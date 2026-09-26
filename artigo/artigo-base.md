# Artigo base

**Rascunho anterior.** A leitura atual do estudo está em `artigo/artigo-v2.md`. Este arquivo guarda a primeira redação. A pergunta ampla, a promessa de eficiência energética e as figuras conceituais daqui não descrevem o que foi medido.

O registro operacional está em `protocol/measurement-protocol.md` e `protocol/experiments.yaml`. As figuras da primeira corrida estão no `README.md`.

---

# Especificação Informativa

Comparativo e Performance de Modelos de IA

## Identificação do projeto

| Campo | Conteúdo |
|---|---|
| Nome do projeto | Comparativo de Performance de Modelos de IA em GPU vs. NPU vs. CPU |
| Autor | Marlon Zanardi |
| Responsáveis | Marlon Zanardi |

## Base de informações e fontes da pesquisa

| Informação | Fonte | Tipo | Info/Tec |
|---|---|---|---|
| CPU é uma arquitetura de propósito geral | Artigo acadêmico | Acadêmica | 2.1 |
| GPUs possuem alta capacidade de processamento paralelo | Artigo + NVIDIA | Acadêmica/Técnica | 2.2 |
| NPUs são especializadas em cargas de IA | Artigo IEEE | Acadêmica | 2.3 |
| Eficiência energética depende do workload | Revisão 2026 | Acadêmica | 2.3 / metodologia |
| Software influencia o desempenho | Revisão 2026 | Acadêmica | Metodologia |
| Não existe um benchmark universal para todos os NPUs | Revisão 2026 | Acadêmica | Metodologia |
| CPU/GPU/NPU podem ter vantagens diferentes conforme a tarefa | Benchmark Edge AI | Experimental | Discussão |

## Objetivo da pesquisa

A presente pesquisa tem como objetivo realizar uma análise comparativa do desempenho de diferentes arquiteturas de processamento utilizadas na execução de modelos de Inteligência Artificial, com foco em CPU (Central Processing Unit), GPU (Graphics Processing Unit) e NPU (Neural Processing Unit). A proposta consiste em investigar como essas arquiteturas se comportam diante de cargas de trabalho de IA, considerando não apenas o tempo necessário para a execução dos modelos, mas também aspectos relacionados à utilização dos recursos computacionais e à eficiência energética.

A pesquisa busca compreender de que maneira as características de cada arquitetura influenciam a execução de modelos de Inteligência Artificial e se determinadas arquiteturas apresentam vantagens ou limitações conforme o tipo de processamento realizado. Dessa forma, o estudo não parte da premissa de que uma arquitetura seja necessariamente superior às demais, mas procura identificar, por meio de uma comparação controlada, como diferentes soluções de hardware respondem a uma mesma carga de trabalho.

Para isso, em uma etapa experimental posterior, serão definidos modelos de Inteligência Artificial e condições de execução que permitam realizar medições comparáveis entre CPU, GPU e NPU. Entre os aspectos considerados estarão o tempo de inferência, utilização dos recursos computacionais e consumo de energia, possibilitando relacionar desempenho e eficiência durante a execução dos modelos.

Com base nos resultados obtidos, pretende-se contribuir para uma compreensão mais clara das características e dos diferentes cenários de utilização dessas arquiteturas, fornecendo uma análise fundamentada sobre sua adequação à execução de modelos de Inteligência Artificial.

## 1. Introdução

A Inteligência Artificial (IA) tornou-se uma das áreas de maior relevância no desenvolvimento tecnológico contemporâneo, estando presente em aplicações que envolvem análise de dados, reconhecimento de padrões, processamento de linguagem e geração de conteúdo. A evolução dessas aplicações tem ampliado tanto as possibilidades de utilização da IA quanto os requisitos necessários para que seus modelos possam ser desenvolvidos e executados de forma eficiente.

Esse avanço estabelece uma relação cada vez mais próxima entre os modelos de Inteligência Artificial e os recursos computacionais responsáveis por sua execução. A capacidade de processamento, a eficiência no uso dos recursos e as características do hardware utilizado podem influenciar diretamente a forma como um determinado modelo é executado. Dessa maneira, compreender essa relação é importante para avaliar as diferentes alternativas disponíveis para processamento de IA.

Entre essas alternativas encontram-se diferentes arquiteturas de processamento, cada uma desenvolvida com características e objetivos específicos. CPU, GPU e NPU representam soluções distintas que podem ser empregadas na execução de aplicações de Inteligência Artificial. Embora possam participar de uma mesma aplicação, suas características arquiteturais fazem com que apresentem comportamentos diferentes conforme as operações e condições de execução.

Diante disso, surge a necessidade de compreender não apenas as características individuais dessas arquiteturas, mas também como elas se comportam quando submetidas a uma mesma tarefa de Inteligência Artificial. Uma comparação experimental permite observar essas diferenças por meio de métricas objetivas, possibilitando relacionar o desempenho obtido com a utilização dos recursos computacionais e a eficiência energética.

Assim, este trabalho propõe uma análise comparativa entre CPU, GPU e NPU na execução de modelos de Inteligência Artificial. Para estabelecer a base necessária à análise, inicialmente serão apresentados o contexto da evolução da computação para IA e os fundamentos relacionados às três arquiteturas. Posteriormente, será realizada uma avaliação experimental utilizando condições controladas e métricas definidas para a comparação dos resultados.

### 1.1 Contexto

O desenvolvimento da computação voltada à Inteligência Artificial está diretamente relacionado à evolução dos próprios modelos computacionais. À medida que novas técnicas passaram a permitir o processamento de maiores volumes de dados e a construção de modelos mais complexos, aumentou também a quantidade de operações necessárias para sua execução. Esse processo fez com que a capacidade de processamento deixasse de ser apenas uma característica de suporte e passasse a representar um dos fatores relevantes para a evolução das aplicações de IA.

Essa transformação também modificou a forma como os sistemas computacionais são projetados para atender às cargas de trabalho de Inteligência Artificial. Em vez de depender exclusivamente de processadores de propósito geral, diferentes arquiteturas passaram a ser empregadas de acordo com as características das operações realizadas. Levantamentos sobre aceleradores de aprendizado de máquina mostram a utilização de CPUs, GPUs e arquiteturas especializadas como alternativas para diferentes necessidades de processamento, considerando aspectos como desempenho, paralelismo e utilização de recursos (REUTHER et al., 2019).

A expansão dessas alternativas está relacionada à necessidade de obter maior eficiência na execução dos modelos. As GPUs, por exemplo, ganharam destaque em aplicações de IA devido à sua capacidade de executar operações de forma altamente paralela, enquanto arquiteturas especializadas foram desenvolvidas para atender de maneira mais direcionada às operações características das redes neurais. Pesquisas sobre aceleradores de redes neurais destacam que fatores como velocidade, throughput, consumo de recursos e eficiência são importantes para avaliar essas diferentes soluções (MOHAIDAT; KHALIL, 2024).

Outro aspecto que passou a receber maior atenção é o custo energético associado à computação de Inteligência Artificial. O aumento da capacidade de processamento necessário para executar modelos modernos também torna relevante analisar quanto de energia é utilizado para obter determinado nível de desempenho. Dessa forma, a avaliação de hardware para IA passa a envolver uma relação entre capacidade de processamento e eficiência, e não somente a velocidade de execução.

Nesse cenário, torna-se necessário compreender como diferentes arquiteturas se comportam diante de uma mesma carga de trabalho. A comparação entre CPU, GPU e NPU pode contribuir para identificar diferenças de desempenho e eficiência e, principalmente, compreender em quais condições determinadas características arquiteturais podem representar vantagens. Essa perspectiva fundamenta a realização da etapa experimental desta pesquisa, na qual essas arquiteturas serão avaliadas utilizando critérios e condições de execução previamente definidos.

### 1.2 Motivação

A evolução das aplicações de Inteligência Artificial tem ampliado a necessidade de recursos computacionais capazes de executar modelos cada vez mais complexos de maneira eficiente. Esse cenário não envolve apenas o aumento da capacidade de processamento, mas também questões relacionadas ao consumo de energia, à utilização dos recursos disponíveis e às características específicas das cargas de trabalho. Dessa forma, a escolha da arquitetura de processamento passa a ser um aspecto relevante no desenvolvimento e na execução de aplicações baseadas em Inteligência Artificial.

Historicamente, as GPUs assumiram uma posição de destaque no processamento de cargas de trabalho relacionadas à Inteligência Artificial devido à sua capacidade de realizar grandes quantidades de operações de forma paralela. Entretanto, a evolução do hardware também trouxe alternativas especializadas, como as NPUs, desenvolvidas especificamente para acelerar determinadas operações características de redes neurais. Ao mesmo tempo, as CPUs continuam desempenhando um papel fundamental nos sistemas computacionais devido à sua flexibilidade e capacidade de executar diferentes tipos de tarefas. A existência dessas arquiteturas com características distintas torna inadequada uma análise baseada exclusivamente na capacidade teórica de processamento de cada componente.

Outro fator relevante está relacionado à eficiência energética. O desempenho de um sistema não deve ser analisado somente pela velocidade com que determinada tarefa é concluída, mas também pela quantidade de recursos necessários para alcançar esse desempenho. Em aplicações de Inteligência Artificial, especialmente aquelas executadas continuamente ou em dispositivos com limitações energéticas, uma arquitetura capaz de realizar determinada tarefa utilizando menos energia pode apresentar vantagens significativas, mesmo que não apresente o maior desempenho absoluto.

Além disso, o comportamento de uma arquitetura pode variar de acordo com o modelo de Inteligência Artificial utilizado e com o tipo de operação executada. Modelos menores, aplicações de inferência e cargas de trabalho destinadas a dispositivos de borda podem apresentar requisitos diferentes daqueles encontrados no treinamento de modelos de grande porte. Consequentemente, uma arquitetura que apresenta vantagens em determinado cenário não necessariamente apresentará o mesmo comportamento em outra condição de execução.

Nesse contexto, torna-se relevante realizar uma comparação experimental entre CPU, GPU e NPU utilizando condições controladas e métricas comuns. Estudos sobre aceleradores de aprendizado de máquina destacam a importância de avaliar diferentes arquiteturas considerando aspectos como desempenho, consumo de recursos, throughput e eficiência, uma vez que esses fatores podem variar de acordo com a aplicação e com a arquitetura empregada (REUTHER et al., 2019; MOHAIDAT; KHALIL, 2024).

A motivação desta pesquisa, portanto, está na necessidade de compreender de forma objetiva como essas três arquiteturas se comportam diante de cargas de trabalho de Inteligência Artificial. A análise proposta busca ultrapassar uma comparação baseada apenas em especificações técnicas, utilizando posteriormente medições experimentais para relacionar o tempo de execução, a utilização dos recursos computacionais e o consumo energético.

Dessa maneira, o estudo pretende contribuir para uma compreensão mais clara das características e limitações de CPU, GPU e NPU, permitindo avaliar em quais condições cada arquitetura pode apresentar vantagens. Essa abordagem também possibilita estabelecer uma base para discutir a adequação de diferentes soluções de hardware às necessidades de aplicações de Inteligência Artificial, sem assumir previamente que uma arquitetura seja superior às demais em todos os cenários.

### 1.3 Questão de Pesquisa

A existência de diferentes arquiteturas de processamento voltadas à execução de aplicações de Inteligência Artificial estabelece um problema relacionado à escolha da solução computacional mais adequada para cada tipo de carga de trabalho. Embora CPU, GPU e NPU possam ser utilizadas na execução de modelos de IA, suas características arquiteturais e formas de processamento são distintas, o que pode resultar em diferentes níveis de desempenho e eficiência.

Diante desse cenário, torna-se necessário avaliar essas arquiteturas sob condições que permitam uma comparação objetiva. A análise deve considerar não apenas o tempo necessário para executar determinado modelo, mas também a utilização dos recursos computacionais e o consumo energético associado à execução. Essa abordagem permite observar o comportamento de cada arquitetura a partir de diferentes aspectos, evitando que a avaliação seja baseada exclusivamente em uma única métrica de desempenho.

Assim, a questão central que orienta esta pesquisa é:

Como as arquiteturas de CPU, GPU e NPU se comportam na execução de modelos de Inteligência Artificial, considerando desempenho, utilização dos recursos computacionais e eficiência energética?

A partir dessa questão, a pesquisa será conduzida por meio de uma etapa experimental na qual modelos de Inteligência Artificial serão executados nas arquiteturas selecionadas sob condições controladas. Os resultados obtidos serão posteriormente analisados e comparados utilizando métricas previamente estabelecidas, permitindo verificar diferenças de comportamento entre as soluções avaliadas.

A questão de pesquisa não pressupõe que uma das arquiteturas apresente desempenho superior em todos os cenários. Ao contrário, busca-se investigar como as características de cada solução influenciam os resultados obtidos diante de diferentes cargas de trabalho, possibilitando relacionar o desempenho observado às condições de execução e aos recursos empregados.

### 1.4 Objetivos Específicos

A partir da questão de pesquisa estabelecida, o presente estudo será desenvolvido por meio de etapas que permitam caracterizar, executar e comparar diferentes arquiteturas de processamento aplicadas a modelos de Inteligência Artificial. Para isso, são definidos os seguintes objetivos específicos:

- Caracterizar as arquiteturas de CPU, GPU e NPU, destacando seus princípios de funcionamento, características de processamento e aplicações relacionadas à Inteligência Artificial;
- Selecionar modelos de Inteligência Artificial que possibilitem a realização de testes comparáveis entre as arquiteturas avaliadas, considerando diferentes características de carga de trabalho;
- Definir condições experimentais padronizadas para a execução dos modelos, buscando reduzir fatores externos que possam comprometer a comparação entre os resultados;
- Medir o desempenho das arquiteturas durante a execução dos modelos, considerando métricas como tempo de inferência e capacidade de processamento;
- Avaliar a utilização dos recursos computacionais e o consumo energético associados à execução das cargas de trabalho selecionadas;
- Comparar os resultados obtidos entre CPU, GPU e NPU, identificando diferenças de comportamento conforme o modelo e as condições de execução;
- Analisar a relação entre desempenho e eficiência energética das arquiteturas avaliadas, considerando que uma maior capacidade de processamento não implica necessariamente maior eficiência;
- Discutir os cenários nos quais as características observadas podem favorecer a utilização de uma determinada arquitetura para aplicações de Inteligência Artificial.

Esses objetivos estabelecem as etapas necessárias para responder à questão de pesquisa, mantendo a análise experimental aberta aos resultados que serão obtidos. Dessa forma, as conclusões sobre o comportamento das arquiteturas serão fundamentadas nas medições realizadas ao longo do estudo, e não em uma classificação previamente estabelecida entre CPU, GPU e NPU.

## 2. Fundamentação Teórica

A compreensão das diferenças de desempenho entre CPU, GPU e NPU exige, inicialmente, o conhecimento das características que definem cada uma dessas arquiteturas. Embora todas possam participar do processamento de aplicações de Inteligência Artificial, suas estruturas internas, formas de execução das operações e níveis de especialização são diferentes. Essas características determinam como cada solução pode lidar com determinados tipos de carga de trabalho.

A CPU, concebida como uma unidade de processamento de propósito geral, prioriza flexibilidade e capacidade de executar diferentes tipos de instruções e aplicações. A GPU apresenta uma organização voltada ao processamento paralelo de grandes quantidades de operações, característica que contribuiu para sua ampla utilização em aplicações de Inteligência Artificial. A NPU, por outro lado, representa uma abordagem mais especializada, sendo desenvolvida para acelerar operações associadas a modelos de redes neurais e outras cargas de trabalho de IA.

Essa distinção entre propósito geral, processamento paralelo e especialização é importante para compreender por que diferentes arquiteturas podem apresentar comportamentos distintos diante de uma mesma tarefa. A avaliação de aceleradores para aprendizado de máquina envolve fatores como desempenho, throughput, consumo de recursos e eficiência, e esses fatores podem variar de acordo com a arquitetura e com as características da carga de trabalho analisada (REUTHER et al., 2019; MOHAIDAT; KHALIL, 2024).

Visão conceitual das arquiteturas de processamento analisadas no estudo.

A Figura 1 apresenta uma visão conceitual da relação entre os modelos de Inteligência Artificial e as três arquiteturas analisadas neste trabalho. A representação destaca que CPU, GPU e NPU podem ser utilizadas como alternativas para a execução de cargas de trabalho de IA, porém possuem características distintas de processamento. Essas diferenças constituem a base para a análise individual de cada arquitetura e, posteriormente, para a comparação experimental proposta pela pesquisa.

A fundamentação teórica será organizada, portanto, a partir da análise individual dessas três soluções. Inicialmente, será apresentada a CPU, abordando seu funcionamento como processador de propósito geral, suas características arquiteturais e sua participação em aplicações de Inteligência Artificial. Em seguida, será analisada a GPU, com ênfase em sua capacidade de processamento paralelo e nas características que favoreceram sua utilização em cargas de trabalho de alta intensidade computacional. Por fim, será apresentada a NPU, considerando seu caráter especializado e sua aplicação na aceleração de operações relacionadas à Inteligência Artificial.

Após essa análise individual, será realizada uma comparação conceitual entre as arquiteturas. Essa etapa permitirá organizar as principais características apresentadas ao longo do capítulo e estabelecer uma relação entre os diferentes modelos de processamento. A comparação conceitual servirá como base para a definição e interpretação da etapa experimental, na qual serão utilizadas métricas objetivas para avaliar o comportamento das arquiteturas.

### 2.1 CPU: Processador de Propósito Geral

A Central Processing Unit (CPU), ou Unidade Central de Processamento, é um componente de propósito geral responsável pela execução de instruções e pelo processamento de diferentes tipos de tarefas em um sistema computacional. Sua principal característica é a flexibilidade, permitindo que o mesmo processador seja utilizado em aplicações com necessidades computacionais distintas, desde atividades de uso geral até tarefas relacionadas à Inteligência Artificial.

As CPUs modernas são compostas por múltiplos núcleos de processamento e utilizam diferentes mecanismos para aumentar a eficiência da execução das instruções. Entre esses mecanismos está a hierarquia de memória, que inclui diferentes níveis de cache posicionados próximos aos núcleos de processamento. Essa organização reduz a necessidade de acessar continuamente a memória principal e pode contribuir para uma execução mais eficiente das aplicações.

Representação conceitual da organização de uma CPU e do fluxo de execução de instruções.

A Figura apresenta uma representação simplificada da organização de uma CPU, destacando a relação entre memória principal, cache, núcleos de processamento e unidades de execução. O fluxo apresentado demonstra, de forma conceitual, como instruções e dados são disponibilizados ao processador durante a execução das operações.

Principais características da CPU consideradas no contexto deste estudo.

| Característica | Descrição |
|---|---|
| Propósito | Processamento de uso geral |
| Flexibilidade | Execução de diferentes tipos de instruções e aplicações |
| Processamento | Execução de instruções por múltiplos núcleos |
| Memória | Utilização de diferentes níveis de cache e memória principal |
| Aplicações em IA | Execução de modelos, preparação de dados e gerenciamento da aplicação |
| Principal característica | Versatilidade e capacidade de adaptação a diferentes cargas de trabalho |
| Limitação em IA | Pode apresentar menor eficiência em determinadas cargas altamente paralelas |

No contexto da Inteligência Artificial, a CPU pode executar diretamente modelos de IA ou atuar como suporte para outras unidades de processamento. Sua flexibilidade permite atender diferentes etapas de uma aplicação, incluindo preparação dos dados, controle da execução e tarefas que não apresentam elevado grau de paralelismo.

Entretanto, determinados modelos de Inteligência Artificial realizam grandes quantidades de operações matemáticas semelhantes, como multiplicações de matrizes e circunvoluções. Essas características podem favorecer arquiteturas desenvolvidas para explorar níveis mais elevados de paralelismo. Dessa forma, a adequação da CPU para uma determinada aplicação de IA depende das características do modelo, da implementação utilizada e da carga de trabalho analisada.

Entre suas principais vantagens estão a flexibilidade, a ampla compatibilidade com diferentes softwares e a capacidade de atuar em diversas funções dentro de um sistema computacional. Como limitação, sua arquitetura de propósito geral pode apresentar menor eficiência em determinados cenários de processamento intensivamente paralelo quando comparada a arquiteturas especializadas para essas operações.

Assim, a CPU representa uma importante referência para a comparação proposta neste trabalho. Sua capacidade de processamento geral permite avaliar como uma arquitetura flexível se comporta diante das mesmas cargas de trabalho que posteriormente serão executadas em arquiteturas com maior especialização, como GPU e NPU.

### 2.2 GPU: Processamento Paralelo de Alto Desempenho

A Graphics Processing Unit (GPU), ou Unidade de Processamento Gráfico, é uma arquitetura de processamento caracterizada pela capacidade de executar grandes quantidades de operações de forma paralela. Embora tenha sido inicialmente desenvolvida para o processamento gráfico, sua organização arquitetural também se mostrou adequada a diferentes aplicações de computação de alto desempenho, incluindo cargas de trabalho relacionadas à Inteligência Artificial.

A principal diferença entre a GPU e um processador de propósito geral está na forma como os recursos computacionais são organizados e utilizados. Enquanto a CPU prioriza flexibilidade e a execução de diferentes tipos de instruções, a GPU disponibiliza um grande número de unidades de processamento capazes de trabalhar simultaneamente sobre diferentes partes de uma tarefa. Esse comportamento permite explorar o paralelismo presente em determinados algoritmos e conjuntos de dados.

Representação conceitual do processamento paralelo em uma GPU.

A Figura representa uma situação em que uma tarefa maior é dividida em diferentes operações que podem ser executadas simultaneamente. Em vez de processar cada operação de maneira individual, a GPU distribui partes da carga entre diferentes unidades de processamento, permitindo que várias operações ocorram ao mesmo tempo. Essa organização é especialmente relevante quando existe grande quantidade de cálculos independentes ou com comportamento semelhante.

O paralelismo, entretanto, não garante automaticamente maior desempenho em qualquer aplicação. Para que os recursos disponíveis sejam aproveitados de maneira eficiente, a carga de trabalho precisa apresentar características que permitam sua divisão em operações executáveis simultaneamente. Além disso, fatores como movimentação de dados, acesso à memória e implementação do software também podem influenciar o resultado obtido.

#### 2.2.1 GPU na execução de Inteligência Artificial

A utilização de GPUs em Inteligência Artificial está relacionada principalmente à natureza de diversas operações presentes em modelos de aprendizado de máquina. Redes neurais podem executar grandes quantidades de operações matemáticas sobre conjuntos de dados, incluindo multiplicações de matrizes e convoluções, que apresentam oportunidades significativas de paralelização.

Essa característica contribuiu para que as GPUs fossem amplamente adotadas em aplicações de treinamento e inferência de modelos de Inteligência Artificial. Estudos sobre aceleradores de aprendizado de máquina destacam o paralelismo como um dos fatores centrais para o desempenho dessas arquiteturas em determinadas cargas de trabalho (REUTHER et al., 2019; MOHAIDAT; KHALIL, 2024).

Além da capacidade computacional, o ecossistema de software também possui importância nesse cenário. Bibliotecas, frameworks e ferramentas desenvolvidas para aproveitar recursos das GPUs permitem que aplicações de Inteligência Artificial distribuam determinadas operações entre as unidades de processamento disponíveis. Dessa forma, o desempenho observado resulta não apenas das características físicas do hardware, mas também da forma como o software utiliza esses recursos.

A GPU também pode apresentar limitações. Cargas de trabalho pequenas, pouco paralelizáveis ou com forte dependência entre operações podem não aproveitar completamente sua capacidade de processamento. Nesses casos, parte dos recursos disponíveis pode permanecer subutilizada, reduzindo a vantagem esperada em relação a outras arquiteturas.

No contexto desta pesquisa, a GPU será analisada como uma arquitetura orientada ao processamento paralelo. Sua avaliação permitirá verificar, experimentalmente, em que medida essa característica contribui para o desempenho e para a eficiência na execução dos modelos selecionados. A comparação com CPU e NPU será realizada posteriormente sob condições controladas, evitando estabelecer antecipadamente qual arquitetura apresentará o melhor resultado.

### 2.3 NPU: Acelerador Especializado para Inteligência Artificial

A Neural Processing Unit (NPU), ou Unidade de Processamento Neural, é uma arquitetura de processamento desenvolvida especificamente para acelerar operações associadas à Inteligência Artificial. Diferentemente de uma CPU, que possui propósito geral, e de uma GPU, cuja principal característica está relacionada ao processamento paralelo de diferentes tipos de operações, a NPU é projetada com foco em cargas de trabalho características de modelos de redes neurais.

A especialização da NPU está relacionada à forma como os modelos de Inteligência Artificial realizam seus cálculos. Operações envolvendo matrizes, tensores e outras transformações matemáticas podem ser tratadas por unidades de processamento estruturadas para executar esse tipo de carga de maneira eficiente. Essa abordagem busca reduzir o custo computacional e energético associado à execução de determinados modelos.

Representação conceitual do fluxo de inferência em uma NPU.

A Figura apresenta, de forma simplificada, o fluxo de uma aplicação de Inteligência Artificial desde a entrada dos dados até a obtenção da saída do modelo. Os dados são inicialmente preparados e encaminhados para as operações necessárias ao modelo, que são processadas pela unidade especializada. Após a execução das operações, o resultado é disponibilizado para a aplicação.

Uma das principais características associadas às NPUs é a busca por eficiência durante a execução de inferência. Em aplicações que precisam executar modelos localmente, como dispositivos móveis, computadores pessoais e outros sistemas de borda, a redução do consumo energético pode ser tão relevante quanto a velocidade absoluta de processamento. Dessa forma, a especialização da NPU está relacionada não apenas ao aumento do desempenho em determinadas operações, mas também à tentativa de realizar essas operações utilizando os recursos de maneira mais eficiente.

#### 2.3.1 NPU na execução de Inteligência Artificial

As NPUs possuem especial importância em cenários nos quais a Inteligência Artificial precisa ser executada diretamente no dispositivo. Esse modelo de processamento está associado ao conceito de Edge AI, no qual parte das tarefas de IA é realizada localmente, reduzindo a necessidade de enviar continuamente os dados para uma infraestrutura remota.

Esse comportamento pode ser relevante em aplicações que exigem menor latência, processamento local ou maior controle sobre os dados. Exemplos incluem reconhecimento de imagens, processamento de áudio, recursos de visão computacional e outras aplicações que utilizam modelos de inferência.

Entretanto, a especialização da NPU também impõe limitações. Como sua arquitetura é desenvolvida para determinados tipos de operações, sua flexibilidade pode ser inferior à encontrada em uma CPU e, em alguns casos, em uma GPU. Além disso, o aproveitamento de seus recursos depende do suporte oferecido pelo sistema operacional, pelos frameworks e pelas ferramentas disponibilizadas pelo fabricante.

Outro fator importante é que o desempenho de uma NPU não pode ser analisado somente com base na existência de um acelerador dedicado. A eficiência observada depende do modelo utilizado, das operações suportadas, da implementação do software e da forma como os dados são transferidos entre os componentes do sistema. Por essa razão, comparações entre diferentes arquiteturas precisam considerar as condições reais de execução.

No contexto deste estudo, a NPU será analisada como uma alternativa especializada para processamento de Inteligência Artificial. Sua avaliação permitirá verificar como essa especialização se relaciona com o tempo de execução, a utilização dos recursos e o consumo energético, possibilitando uma comparação direta com as características observadas nas CPUs e GPUs.

Assim, a NPU representa uma abordagem distinta dentro do processamento de IA: em vez de priorizar a flexibilidade de um processador de propósito geral ou o paralelismo amplo de uma GPU, busca utilizar uma arquitetura direcionada às operações mais características dos modelos de Inteligência Artificial. Essa diferença será relevante para a etapa experimental, na qual seu comportamento será observado sob condições controladas.

### 2.4 Comparação Conceitual entre CPU, GPU e NPU

A análise individual das arquiteturas permite identificar que CPU, GPU e NPU foram desenvolvidas a partir de diferentes prioridades arquiteturais. A CPU apresenta maior flexibilidade para executar tarefas variadas, a GPU explora de maneira intensa o paralelismo disponível em determinadas cargas de trabalho e a NPU direciona seus recursos para operações características da Inteligência Artificial. Essas diferenças não significam que uma arquitetura seja universalmente superior às demais, mas indicam que suas características podem ser mais adequadas a determinados tipos de processamento.

Para organizar essas diferenças, a Tabela 3 apresenta uma comparação conceitual das principais características discutidas nas sessões anteriores.

Comparação conceitual entre CPU, GPU e NPU.

| Característica | CPU | GPU | NPU |
|---|---|---|---|
| Objetivo arquitetural | Propósito geral | Processamento altamente paralelo | Aceleração especializada de IA |
| Flexibilidade | Alta | Alta, com foco em cargas paralelizáveis | Mais limitada |
| Paralelismo | Presente, porém associado a uma arquitetura geral | Elevado | Elevado e direcionado a operações de IA |
| Especialização em IA | Baixa | Alta | Muito alta |
| Principais aplicações | Uso geral, controle e processamento diversificado | Computação paralela, treinamento e inferência | Inferência e aplicações de IA local |
| Eficiência em cargas de IA | Dependente da carga | Dependente do grau de paralelismo | Direcionada à eficiência em operações suportadas |
| Flexibilidade de software | Muito ampla | Ampla, dependendo do ecossistema | Dependente do suporte do fabricante e software |
| Ponto de destaque | Versatilidade | Capacidade de processamento paralelo | Especialização |
| Principal limitação | Menor adequação a algumas cargas altamente paralelas | Pode ser subutilizada em cargas pequenas ou pouco paralelizáveis | Menor flexibilidade para cargas não suportadas |

A comparação demonstra que a escolha de uma arquitetura não deve considerar apenas sua capacidade máxima de processamento. Uma CPU pode apresentar vantagens em tarefas que exigem flexibilidade e diferentes tipos de instruções, enquanto uma GPU pode aproveitar melhor cargas compostas por grande quantidade de operações independentes. A NPU, por sua vez, pode ser particularmente adequada quando o processamento está concentrado em operações de Inteligência Artificial compatíveis com sua arquitetura.

Relação conceitual entre tipo de carga de trabalho e características das arquiteturas CPU, GPU e NPU.

A Figura apresenta essa relação de forma conceitual. Em vez de estabelecer uma classificação absoluta de desempenho, o esquema relaciona três características principais — flexibilidade, paralelismo e especialização — às arquiteturas analisadas. Essa representação reforça que diferentes tipos de carga podem explorar características distintas do hardware.

A comparação também evidencia a importância do software no aproveitamento das arquiteturas. A existência de um determinado recurso de hardware não garante, isoladamente, que uma aplicação obterá o máximo desempenho possível. Bibliotecas, frameworks, compiladores, drivers e mecanismos de suporte à aceleração influenciam a forma como os recursos disponíveis são utilizados.

Por essa razão, as diferenças apresentadas nesta seção devem ser entendidas como características arquiteturais e não como resultados experimentais. A verificação do desempenho efetivo de CPU, GPU e NPU será realizada posteriormente, utilizando modelos, condições de execução e métricas definidos na metodologia.

A fundamentação apresentada até este ponto estabelece, portanto, a base necessária para a etapa experimental. Ao compreender as diferenças entre propósito geral, processamento paralelo e especialização em IA, torna-se possível interpretar de maneira mais adequada os resultados que serão obtidos durante os testes.

## Medição

Os números da medição não ficam neste rascunho. O manuscrito que os arquivos sustentam é `artigo/artigo-v2.md`. As sessões citadas são `results/windows-workstation/20260926T195237Z` e `results/apple-m4-pro/20260926T204025Z`. A sessão do M1 fica fora dessas figuras.
