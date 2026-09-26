#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <math.h>
#import <time.h>

static void fail(NSString *message) __attribute__((noreturn));

static void fail(NSString *message) {
    fprintf(stderr, "%s\n", message.UTF8String);
    exit(1);
}

static MLComputeUnits unitsFromName(NSString *name) {
    if ([name isEqualToString:@"cpuOnly"]) {
        return MLComputeUnitsCPUOnly;
    }
    if ([name isEqualToString:@"cpuAndGPU"]) {
        return MLComputeUnitsCPUAndGPU;
    }
    if ([name isEqualToString:@"cpuAndNeuralEngine"]) {
        return MLComputeUnitsCPUAndNeuralEngine;
    }
    if ([name isEqualToString:@"all"]) {
        return MLComputeUnitsAll;
    }
    fail([NSString stringWithFormat:@"unknown compute units: %@", name]);
}

static NSString *deviceName(id<MLComputeDeviceProtocol> device, NSNumber **neuralCores) {
    if ([device isKindOfClass:[MLCPUComputeDevice class]]) {
        return @"cpu";
    }
    if ([device isKindOfClass:[MLGPUComputeDevice class]]) {
        return @"gpu";
    }
    if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
        *neuralCores = @(((MLNeuralEngineComputeDevice *)device).totalCoreCount);
        return @"neuralEngine";
    }
    return @"unknown";
}

static NSString *dataTypeName(MLMultiArrayDataType dataType) {
    switch (dataType) {
        case MLMultiArrayDataTypeFloat16:
            return @"float16";
        case MLMultiArrayDataTypeFloat32:
            return @"float32";
        case MLMultiArrayDataTypeFloat64:
            return @"float64";
        case MLMultiArrayDataTypeInt32:
            return @"int32";
        default:
            return @"other";
    }
}

static void addCount(NSMutableDictionary<NSString *, NSNumber *> *counts, NSString *key) {
    counts[key] = @(counts[key].integerValue + 1);
}

static void addWeight(NSMutableDictionary<NSString *, NSNumber *> *weights, NSString *key, double value) {
    weights[key] = @(weights[key].doubleValue + value);
}

static NSMutableDictionary<NSString *, NSNumber *> *childMap(
    NSMutableDictionary<NSString *, NSMutableDictionary<NSString *, NSNumber *> *> *root,
    NSString *key
) {
    NSMutableDictionary<NSString *, NSNumber *> *child = root[key];
    if (child == nil) {
        child = [NSMutableDictionary dictionary];
        root[key] = child;
    }
    return child;
}

static void walk(
    MLModelStructureProgramBlock *block,
    MLComputePlan *plan,
    NSMutableDictionary<NSString *, NSNumber *> *counts,
    NSMutableDictionary<NSString *, NSNumber *> *weights,
    NSMutableDictionary<NSString *, NSMutableDictionary<NSString *, NSNumber *> *> *operatorCounts,
    NSMutableDictionary<NSString *, NSMutableDictionary<NSString *, NSNumber *> *> *operatorWeights,
    NSNumber **neuralCores
) {
    for (MLModelStructureProgramOperation *operation in block.operations) {
        MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:operation];
        NSString *device = usage == nil ? @"unknown" : deviceName(usage.preferredComputeDevice, neuralCores);
        MLComputePlanCost *cost = [plan estimatedCostOfMLProgramOperation:operation];
        double weight = cost == nil ? 0 : cost.weight;
        addCount(counts, device);
        addWeight(weights, device, weight);
        addCount(childMap(operatorCounts, operation.operatorName), device);
        addWeight(childMap(operatorWeights, operation.operatorName), device, weight);
        for (MLModelStructureProgramBlock *nestedBlock in operation.blocks) {
            walk(nestedBlock, plan, counts, weights, operatorCounts, operatorWeights, neuralCores);
        }
    }
}

static void emit(NSDictionary *payload) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload options:NSJSONWritingSortedKeys error:&error];
    if (data == nil) {
        fail(error.localizedDescription);
    }
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

static MLModelConfiguration *configuration(NSString *computeUnits) {
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = unitsFromName(computeUnits);
    return config;
}

static MLComputePlan *loadPlan(NSURL *url, NSString *computeUnits) {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block MLComputePlan *plan = nil;
    __block NSError *error = nil;
    [MLComputePlan loadContentsOfURL:url
                       configuration:configuration(computeUnits)
                   completionHandler:^(MLComputePlan *computePlan, NSError *loadError) {
                       plan = computePlan;
                       error = loadError;
                       dispatch_semaphore_signal(semaphore);
                   }];
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
    if (plan == nil) {
        fail(error.localizedDescription ?: @"failed to load compute plan");
    }
    return plan;
}

static void reportPlan(NSURL *url, NSString *computeUnits) {
    MLComputePlan *plan = loadPlan(url, computeUnits);
    MLModelStructureProgram *program = plan.modelStructure.program;
    if (program == nil) {
        fail(@"model is not an mlprogram");
    }
    MLModelStructureProgramFunction *mainFunction = program.functions[@"main"];
    if (mainFunction == nil) {
        fail(@"model has no main function");
    }
    NSMutableDictionary<NSString *, NSNumber *> *counts = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSNumber *> *weights = [NSMutableDictionary dictionary];
    NSMutableDictionary *operatorCounts = [NSMutableDictionary dictionary];
    NSMutableDictionary *operatorWeights = [NSMutableDictionary dictionary];
    NSNumber *neuralCores = nil;
    walk(mainFunction.block, plan, counts, weights, operatorCounts, operatorWeights, &neuralCores);
    NSInteger operationCount = 0;
    for (NSNumber *count in counts.allValues) {
        operationCount += count.integerValue;
    }
    NSMutableDictionary *payload = [@{
        @"compute_units": computeUnits,
        @"operation_count": @(operationCount),
        @"operator_counts": operatorCounts,
        @"operator_weight": operatorWeights,
        @"preferred_counts": counts,
        @"preferred_weight": weights,
    } mutableCopy];
    if (neuralCores != nil) {
        payload[@"neural_engine_core_count"] = neuralCores;
    }
    emit(payload);
}

static void fill(MLMultiArray *array) {
    NSInteger count = array.count;
    for (NSInteger index = 0; index < count; index++) {
        array[index] = @((double)(index % 251) / 251.0);
    }
}

static double meanAbs(MLMultiArray *array) {
    NSInteger limit = MIN(array.count, 1024);
    if (limit == 0) {
        return 0;
    }
    double sum = 0;
    for (NSInteger index = 0; index < limit; index++) {
        sum += fabs(array[index].doubleValue);
    }
    return sum / (double)limit;
}

static NSArray<NSNumber *> *shapeNumbers(NSArray<NSNumber *> *shape) {
    NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:shape.count];
    for (NSNumber *dimension in shape) {
        [values addObject:@(dimension.integerValue)];
    }
    return values;
}

static uint64_t unixNanoseconds(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_REALTIME, &stamp);
    return (uint64_t)stamp.tv_sec * 1000000000ull + (uint64_t)stamp.tv_nsec;
}

static MLDictionaryFeatureProvider *preparedInput(NSURL *url, NSString *computeUnits, MLModel **loaded) {
    NSError *error = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:url configuration:configuration(computeUnits) error:&error];
    if (model == nil) {
        fail(error.localizedDescription ?: @"failed to load model");
    }
    NSString *inputName = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
    if (inputName == nil) {
        fail(@"model has no inputs");
    }
    MLMultiArrayConstraint *constraint = model.modelDescription.inputDescriptionsByName[inputName].multiArrayConstraint;
    if (constraint == nil) {
        fail(@"model input is not a multiarray");
    }
    MLMultiArray *array = [[MLMultiArray alloc] initWithShape:constraint.shape dataType:constraint.dataType error:&error];
    if (array == nil) {
        fail(error.localizedDescription ?: @"failed to allocate input");
    }
    fill(array);
    MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inputName: array} error:&error];
    if (provider == nil) {
        fail(error.localizedDescription ?: @"failed to build input");
    }
    *loaded = model;
    return provider;
}

static void measure(NSURL *url, NSString *computeUnits, int warmup, int iterations) {
    if (warmup < 1 || iterations < 1) {
        fail(@"warmup and iterations must be positive");
    }
    NSError *error = nil;
    MLModel *model = nil;
    MLDictionaryFeatureProvider *provider = preparedInput(url, computeUnits, &model);
    NSString *inputName = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
    MLMultiArrayConstraint *constraint = model.modelDescription.inputDescriptionsByName[inputName].multiArrayConstraint;
    NSMutableArray<NSNumber *> *warmupLatencies = [NSMutableArray arrayWithCapacity:warmup];
    double outputMean = 0;
    for (int index = 0; index < warmup; index++) {
        uint64_t started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
        uint64_t finished = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        if (prediction == nil) {
            fail(error.localizedDescription ?: @"prediction failed");
        }
        [warmupLatencies addObject:@((double)(finished - started) / 1e6)];
        NSString *outputName = prediction.featureNames.anyObject;
        MLMultiArray *output = outputName == nil ? nil : [prediction featureValueForName:outputName].multiArrayValue;
        if (output == nil) {
            fail(@"prediction output is not a multiarray");
        }
        outputMean = meanAbs(output);
        if (!isfinite(outputMean)) {
            fail(@"prediction output is not finite");
        }
    }
    uint64_t windowStart = unixNanoseconds();
    NSMutableArray<NSNumber *> *latencies = [NSMutableArray arrayWithCapacity:iterations];
    for (int index = 0; index < iterations; index++) {
        uint64_t started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
        uint64_t finished = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        if (prediction == nil) {
            fail(error.localizedDescription ?: @"prediction failed");
        }
        [latencies addObject:@((double)(finished - started) / 1e6)];
    }
    uint64_t windowEnd = unixNanoseconds();
    emit(@{
        @"compute_units": computeUnits,
        @"input_data_type": dataTypeName(constraint.dataType),
        @"input_name": inputName,
        @"input_shape": shapeNumbers(constraint.shape),
        @"latencies_ms": latencies,
        @"measure_end_unix_ns": [NSString stringWithFormat:@"%llu", windowEnd],
        @"measure_start_unix_ns": [NSString stringWithFormat:@"%llu", windowStart],
        @"output_mean_abs": @(outputMean),
        @"warmup_latencies_ms": warmupLatencies,
    });
}

static void sustain(NSURL *url, NSString *computeUnits, int warmup, int minimumMilliseconds) {
    if (warmup < 1 || minimumMilliseconds < 1) {
        fail(@"warmup and duration must be positive");
    }
    NSError *error = nil;
    MLModel *model = nil;
    MLDictionaryFeatureProvider *provider = preparedInput(url, computeUnits, &model);
    for (int index = 0; index < warmup; index++) {
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
        if (prediction == nil) {
            fail(error.localizedDescription ?: @"prediction failed");
        }
    }
    uint64_t windowStart = unixNanoseconds();
    uint64_t started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    int inferences = 0;
    while ((clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started) / 1000000ull < (uint64_t)minimumMilliseconds) {
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
        if (prediction == nil) {
            fail(error.localizedDescription ?: @"prediction failed");
        }
        inferences += 1;
    }
    uint64_t windowEnd = unixNanoseconds();
    emit(@{
        @"compute_units": computeUnits,
        @"inferences": @(inferences),
        @"measure_end_unix_ns": [NSString stringWithFormat:@"%llu", windowEnd],
        @"measure_start_unix_ns": [NSString stringWithFormat:@"%llu", windowStart],
        @"minimum_milliseconds": @(minimumMilliseconds),
    });
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 4) {
            fail(@"usage: bench <plan|predict> <model.mlmodelc> <cpuOnly|cpuAndGPU|cpuAndNeuralEngine> [warmup iterations]");
        }
        NSString *command = [NSString stringWithUTF8String:argv[1]];
        NSString *path = [NSString stringWithUTF8String:argv[2]];
        NSString *computeUnits = [NSString stringWithUTF8String:argv[3]];
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            fail([NSString stringWithFormat:@"model not found: %@", path]);
        }
        NSURL *url = [NSURL fileURLWithPath:path];
        if ([command isEqualToString:@"plan"]) {
            reportPlan(url, computeUnits);
            return 0;
        }
        if ([command isEqualToString:@"predict"]) {
            if (argc != 6) {
                fail(@"usage: bench predict <model.mlmodelc> <compute-units> <warmup> <iterations>");
            }
            int warmup = atoi(argv[4]);
            int iterations = atoi(argv[5]);
            measure(url, computeUnits, warmup, iterations);
            return 0;
        }
        if ([command isEqualToString:@"sustain"]) {
            if (argc != 6) {
                fail(@"usage: bench sustain <model.mlmodelc> <compute-units> <warmup> <minimum-milliseconds>");
            }
            int warmup = atoi(argv[4]);
            int minimumMilliseconds = atoi(argv[5]);
            sustain(url, computeUnits, warmup, minimumMilliseconds);
            return 0;
        }
        fail([NSString stringWithFormat:@"unknown command: %@", command]);
    }
}
