# Protocolo de medição

Registro para executar o estudo em duas máquinas. Os números de uma máquina não entram no gráfico da outra. Cada corrida produz um manifesto próprio. A comparação de CPU, GPU e NPU existe dentro desse manifesto.

Escopo: inferência. Treino fica fora.

Parâmetros fixos estão em `protocol/experiments.yaml`.

## Primeira sessão no Mac

Confirmar o chip antes de medir qualquer modelo. A corrida registrada em 26 de setembro de 2026 usou um MacBook Pro com Apple M4 Pro. Esse chip tem Neural Engine e entra no estudo. Um Mac Intel não tem Neural Engine e sai.

```bash
system_profiler SPHardwareDataType
sw_vers
sysctl -n machdep.cpu.brand_string
```

Anotar no manifesto: modelo do Mac, chip, núcleos de CPU, núcleos de GPU, memória unificada, versão do macOS.

Confirmar que o sistema expõe potência de CPU, GPU e Neural Engine:

```bash
sudo powermetrics -s cpu_power,gpu_power,ane_power -n 5 -i 1000
```

A amostra precisa listar os três samplers. Se `ane_power` não existir, o Mac não fecha a métrica de energia do Neural Engine.

No M4 Pro medido, a GPU e o Neural Engine são blocos separados. Uma corrida só vale para o dispositivo que o plano de execução realmente usou. No float32 desta corrida, `cpuAndNeuralEngine` ficou na CPU e não entra como NPU.

## Regras das duas máquinas

- O mesmo modelo e a mesma entrada nas duas máquinas: MobileNetV2 e ResNet-50, tensor `1x3x224x224`.
- O teste principal é batch 1. Os batches 8 e 32 existem para mostrar quando o paralelismo da GPU passa a dominar. No Mac, batch 1 é o resultado que responde o cenário de borda.
- Cada sessão descarta 20 iterações de aquecimento e grava 100 iterações. São 3 sessões, com a máquina em idle térmico entre elas.
- O relatório usa mediana e percentil 95. A média de uma sessão única não entra no texto.
- FP32 é a coluna de comparação entre arquiteturas. FP16 na GPU NVIDIA e o caminho quantizado no Neural Engine formam outra coluna, com outro rótulo.
- O dispositivo publicado é o bloco que executou o modelo, verificado por log, não o nome comercial da máquina.

## Experimento A — Windows

Inventário feito em 2026-09-26 nesta workstation:

| Item | Valor |
|---|---|
| CPU | AMD Ryzen 9 7900X3D, 12 núcleos, 24 threads |
| GPU | NVIDIA GeForce RTX 4070 Ti, 12282 MiB, driver 581.42 |
| NPU | ausente |
| GPU integrada | AMD Radeon Graphics, fora do estudo |
| SO | Windows 10.0.26200 |
| Python | 3.11 (`C:\Program Files\Python311\python.exe`) |

Runtime: ONNX Runtime. Dispositivos: `CPUExecutionProvider` e `CUDAExecutionProvider`. A medição de GPU espera a sincronização do dispositivo antes de parar o cronômetro.

Energia: potência da placa lida durante a janela medida e integrada no tempo, em joule por inferência. Esse número é potência de placa. A tomada e o pacote do Ryzen ficam de fora dessa coluna, salvo medição adicional registrada no manifesto.

A Radeon integrada não ocupa o lugar da NPU.

## Experimento B — Mac

Runtime: Core ML. O modelo ONNX do experimento A é convertido, e o manifesto guarda a ferramenta, a versão e o arquivo gerado.

Unidades de computação:

| Corrida | `computeUnits` | O que representa |
|---|---|---|
| CPU | `cpuOnly` | CPU |
| GPU | `cpuAndGPU` | GPU Apple, incluindo Neural Accelerator dos núcleos gráficos |
| NPU | `cpuAndNeuralEngine` | Neural Engine, com a CPU nas operações que ele não absorve |

`all` não é corrida do estudo. Esse modo mistura CPU, GPU e Neural Engine.

Não existe modo somente Neural Engine. A frase do artigo descreve `cpuAndNeuralEngine` com essa ressalva.

Uma corrida de NPU entra na tabela quando as duas condições abaixo se confirmam:

1. O `MLComputePlan` coloca a maioria das operações no Neural Engine.
2. Durante a janela medida, `ane_power` sobe em relação ao idle, e permanece no idle nas corridas `cpuOnly` e `cpuAndGPU`.

Se a maioria das operações cair na CPU, o resultado é fallback e não entra na coluna NPU.

Energia: `powermetrics` com `cpu_power`, `gpu_power` e `ane_power`, amostrado durante a janela, integrado em joule por inferência por trilho. Notebook na tomada, tela em repouso, outros aplicativos fechados. O valor é estimativa do chip, não da tomada.

## Pasta de evidência

```text
results/{machine_id}/{utc_timestamp}/
  manifest.json
  samples.jsonl
  summary.csv
  figures/
```

`manifest.json` leva máquina, chip, SO, runtime, versão, hash do modelo, precisão, `computeUnits` ou execution provider, formato da entrada, warmup, iterações e a origem da potência.

`samples.jsonl` leva uma linha por iteração: sessão, dispositivo, batch, latência e, quando houver, potência amostrada.

As figuras saem desses arquivos. As quatro figuras de cada máquina:

1. Latência em batch 1, por modelo e dispositivo.
2. Throughput por batch.
3. Joule por inferência, com a fonte da potência na legenda.
4. Dispersão de latência por joule.

Figura do Windows e figura do Mac ficam em painéis separados. O texto discute o comportamento de cada arquitetura dentro da sua máquina. O cruzamento entre máquinas usa o ganho contra a CPU da própria máquina, em `results/comparison/20260926/`.

## Medição de 26 de setembro de 2026

Isto é registro do que as duas corridas mostraram. Não é a conclusão do artigo.

| Máquina | Condição que muda a leitura |
|---|---|
| Windows, Ryzen 9 7900X3D e RTX 4070 Ti | ONNX Runtime, FP32 com TF32 desligado, desktop, pausa de 60 s. Sem NPU. Energia só da placa. |
| MacBook Pro M4 Pro, 24 GB | Core ML, na bateria, pausa de 2 s, sem `powermetrics`. Batch 1 e 8. |

No Windows, em FP32 e batch 1, a mediana ficou em 3,04 ms na CPU e 2,38 ms na GPU para o MobileNetV2, e em 32,7 ms na CPU e 3,32 ms na GPU para o ResNet-50. A GPU discreta se separa da CPU quando o modelo pesa ou o batch sobe. O p95 da CPU no ResNet-50 passa de 80 ms. A máquina não estava isolada.

No M4 Pro, a comparação de três dispositivos existe em float16. Em batch 1, o MobileNetV2 ficou em 1,91 ms na CPU, 1,06 ms na GPU e 0,36 ms no Neural Engine. O ResNet-50 ficou em 4,07 ms na CPU e 1,06 ms no Neural Engine. O pedido de GPU para o ResNet-50 em batch 1 ficou majoritariamente na CPU e foi descartado. Em float32 o Neural Engine não executou nenhum dos dois modelos.

Contra a CPU da mesma máquina e na mesma precisão, no batch 1, a 4070 Ti chega a cerca de 9,9 vezes no ResNet-50 em FP32 e fica perto de 1,3 vezes no MobileNetV2. No M4 Pro, em float16, o Neural Engine chega a cerca de 5,3 vezes no MobileNetV2 e 3,8 vezes no ResNet-50. Esses ganhos não ordenam o Neural Engine contra a 4070 Ti.

Energia da tomada e joule da CPU continuam sem medição. O Mac precisa de uma repetição na tomada antes que esses milissegundos sejam citados como a condição cheia do chip.
