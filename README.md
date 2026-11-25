# manometry-back

API que recebe o laudo de uma manometria esofágica de alta resolução, recorta as
topografias de pressão das dez deglutições de água e devolve a classificação do
exame pela **Classificação de Chicago v4.0**, junto das métricas que sustentam a
decisão.

O modelo é o proposto no artigo JBHI de
[`manometry-classification`](../manometry-classification) (`studies/cbm_2026`):
um ConvNeXt-Tiny treinado em multitarefa lê cada deglutição e estima, além do
diagnóstico, os estados e as medidas de Chicago (DCI, IRP, DL, quebra,
pressurização), supervisionados pelos valores do próprio software de manometria.
No nível do exame, uma logística lê os eventos de contagem exatos desses estados
(Poisson-binomial), as frações, a IRP mediana, a compatibilidade com as regras da
v4.0 e o caminho da imagem. Sete classes, partição por paciente.

```
laudo .docx/.rtf
   │  leitura dos PNG na ordem do documento
   ▼
paineis  ──▶ retrato = deglutição · paisagem = logotipo
   │
   │  unifier.images.clean_swallow  (código do construtor do dataset, verbatim)
   ▼
topografia recortada ──┬──▶ 224px ──▶ ConvNeXt-Tiny ──┬──▶ estados + DCI/IRP/DL por deglutição
                       │                              └──▶ diagnóstico por deglutição
                       │                                          │
                       │               eventos de contagem + regras v4.0 + imagem
                       │                                          ▼
                       │                               logística ──▶ 7 classes
                       └──▶ inversão da rampa jet ──▶ quebras, JEG basal (só exibição)
```

## Qual modelo o serviço carrega

O artigo avalia o modelo em 15 partições repetidas de 5 folds (75 folds): macro-F1
0,825 e acurácia 0,928, contra 0,697 e 0,836 da média das probabilidades por
deglutição. Só o encoder da partição 42, fold 0, foi guardado (os outros 74 foram
treinados, extraídos e descartados), então é ele que o serviço carrega.
`studies/cbm_2026/experiments/jbhi_deploy.py` reajusta o readout desse fold
exatamente como `jbhi_fusion.py` faz (logística nos pacientes de treino, C e
temperatura na validação) e grava o bundle em
`studies/cbm_2026/artifacts/jbhi_2026_11/deploy/`. Nos 358 exames de teste desse
fold: macro-F1 0,849, acurácia 0,944.

| classe | F1 no teste do fold | exames de teste |
|---|---:|---:|
| contratilidade ausente | 0,954 | 33 |
| acalásia tipo I | 0,894 | 23 |
| acalásia tipo II | 0,875 | 32 |
| obstrução da saída da JEG | 0,500 | 5 |
| esôfago hipercontrátil | 0,800 | 7 |
| motilidade esofágica ineficaz | 0,944 | 69 |
| motilidade normal | 0,976 | 189 |

Os pesos ficam em meia precisão no disco (56 MB, abaixo do limite de 100 MB do
GitHub) e voltam a float32 na carga. O que o serviço mostra por deglutição (padrão
de contratilidade, DCI, IRP 4s, latência distal) e as portas de Chicago do estudo
saem das cabeças de medida do modelo, as mesmas que o readout consome. Quebras,
pressão basal da JEG e frações de pressurização continuam vindo da inversão da
rampa de cor e são só evidência de apoio: o modelo não as lê.

`tests/test_inference_parity.py` reexecuta os exames de teste do fold pelo caminho
da API, a partir dos PNG do corpus, e exige que as probabilidades batam com as
guardadas no bundle. Nos 358 exames: concordância de 100% no argmax, desvio médio
8,5e-5 (bfloat16 na GPU do treino contra float32 aqui). `tests/test_end_to_end.py`
fecha o circuito pelo HTTP, com laudos montados a partir de exames reais.

## Instalação

```bash
python3 -m pip install -r requirements.txt     # torch conforme a sua CUDA
uvicorn app.main:app --reload                  # http://127.0.0.1:8000/docs
```

O checkpoint e a calibração de cor não são versionados aqui: são artefatos
grandes e derivados de dados de paciente, e vivem em `manometry-classification`.
Com os dois repositórios lado a lado nada precisa ser configurado.

### Variáveis de ambiente

| variável | padrão | para quê |
|---|---|---|
| `MANOMETRY_CLASSIFICATION_ROOT` | `../manometry-classification` | raiz de onde saem checkpoint, calibração e corpus |
| `MANOMETRY_CHECKPOINT` | `<root>/studies/cbm_2026/artifacts/jbhi_2026_11/deploy/model.pt` | outro bundle |
| `MANOMETRY_CHECKPOINT_URL` | vazio | baixa o checkpoint no primeiro boot, se o arquivo local não existir |
| `MANOMETRY_RESULT_JSON` | `result.json` ao lado do checkpoint | métricas publicadas em `/api/model` |
| `MANOMETRY_PREDICTIONS` | `test_predictions.npz` ao lado do checkpoint | desempenho por classe |
| `MANOMETRY_CALIBRATION` | `assets/model/image_chicago_calibration.json` | calibração da rampa de cor |
| `MANOMETRY_DATASET_ROOT` | `<root>/data` | corpus, para imagens de referência e laudos de exemplo |
| `MANOMETRY_REFERENCE_DIR` | `assets/reference` | PNG de referência já exportados |
| `MANOMETRY_SAMPLE_DIR` | `assets/samples` | laudos de demonstração já gerados |
| `MANOMETRY_DEVICE` | `auto` | `cuda`, `cpu` |
| `MANOMETRY_TORCH_THREADS` | `0` (deixa o torch decidir) | limita as threads em container compartilhado |
| `MANOMETRY_LAZY_MODEL` | `true` | `false` carrega o checkpoint no boot, para readiness probe |
| `MANOMETRY_MAX_UPLOAD_BYTES` | `67108864` | limite do upload |
| `MANOMETRY_ALLOWED_ORIGINS` | `localhost:5173,4173` | CORS, lista exata |
| `MANOMETRY_ALLOWED_ORIGIN_REGEX` | vazio | CORS por padrão, para deploys de preview |

## Endpoints

| método | rota | o que faz |
|---|---|---|
| `POST` | `/api/analyze` | laudo `.docx`/`.rtf` para diagnóstico, ranking, métricas por deglutição e portas de Chicago |
| `GET` | `/api/health` | estado do serviço e se o checkpoint está acessível |
| `GET` | `/api/model` | procedência do modelo e desempenho por classe na partição de teste |
| `GET` | `/api/diseases` | texto clínico das sete classes |
| `GET` | `/api/diseases/{slug}/reference.png` | topografia real da classe |
| `GET` | `/api/vocabulary` | rótulos dos padrões de deglutição e das métricas |
| `GET` | `/api/samples` | classes com laudo de exemplo nesta instância |
| `GET` | `/api/sample-report?diagnosis=<slug>` | laudo `.docx` de demonstração |

O arquivo enviado nunca toca o disco: é lido em memória, as imagens são
recortadas, medidas e descartadas com a resposta.

## Deploy no Cloud Run

O `Dockerfile` empacota a API, o bundle do modelo e apenas as dependências de CPU.
O mesmo repositório é implantado diretamente pelo Cloud Build; não é necessário
criar um repositório separado nem manter o Render no caminho da inferência.

Antes do primeiro deploy, instale o [Google Cloud CLI](https://cloud.google.com/sdk/docs/install),
crie ou selecione um projeto com faturamento habilitado e autentique:

```bash
gcloud auth login
gcloud config set project SEU_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Na raiz deste repositório, publique em `southamerica-east1` (São Paulo), para que
os laudos sejam processados no Brasil:

```bash
gcloud run deploy manometry-back \
  --source . \
  --region southamerica-east1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 1 \
  --min 0 \
  --max 1 \
  --timeout 300 \
  --cpu-boost \
  --set-env-vars 'MANOMETRY_CHECKPOINT=assets/model/model.pt,MANOMETRY_RESULT_JSON=assets/model/result.json,MANOMETRY_PREDICTIONS=assets/model/test_predictions.npz,MANOMETRY_CALIBRATION=assets/model/image_chicago_calibration.json,MANOMETRY_DEVICE=cpu,MANOMETRY_TORCH_THREADS=2,MANOMETRY_LAZY_MODEL=false,MANOMETRY_ALLOWED_ORIGINS=https://SEU_FRONT.vercel.app,MANOMETRY_ALLOWED_ORIGIN_REGEX=https://.*\.vercel\.app'
```

Troque `SEU_PROJECT_ID` e `https://SEU_FRONT.vercel.app`. A saída do deploy traz
uma URL `https://...run.app`; teste `GET /api/health` nela e depois configure essa
URL, sem barra final, como `VITE_API_BASE` no projeto do frontend na Vercel.
Como variáveis `VITE_*` entram no bundle durante o build, faça um novo deploy do
frontend depois da troca.

As escolhas acima são intencionais:

- **2 GiB** comportam o pico próximo de 1 GiB do processo;
- **um worker e concorrência 1** impedem inferências simultâneas de disputarem
  CPU e memória dentro da mesma instância;
- **mínimo zero** preserva o scale-to-zero e a cota gratuita;
- **máximo um** evita custo e uso de memória multiplicados durante a validação;
- **startup CPU boost** reduz o tempo de importação do PyTorch e carga do modelo;
- **modelo não lazy** faz a instância ficar pronta apenas depois de carregar o
  checkpoint.

São Paulo tem preço por segundo mais alto que `us-central1`, o que pode consumir
a cota gratuita de forma diferente; com uso esporádico continua perto de zero.
O repositório de imagens do Artifact Registry tem política de limpeza que mantém
só as duas versões mais recentes, para caber nos 0,5 GB gratuitos. Orçamentos do Google Cloud apenas alertam: não funcionam como teto de
gastos. Mantenha `--max 1`, configure alertas de faturamento e acompanhe o uso.

Em produção o serviço responde em
`https://manometry-back-273216239564.southamerica-east1.run.app`, que é o
`VITE_API_BASE` do frontend na Vercel.

## Deploy no Render

`render.yaml` já traz o blueprint. O que decide se funciona:

**Memória.** O processo chega perto de **1,1 GB** de pico, somando o runtime do
torch, os pesos do ConvNeXt-Tiny e as ativações de dez imagens. O plano é o
**Standard (2 GB)**; os de 512 MB não sustentam.

**O arquivo de pesos.** O bundle tem 56 MB (pesos em meia precisão) e fica
versionado em `assets/model/model.pt`, abaixo do limite de 100 MB por arquivo do
GitHub. Isso deixa o deploy autocontido e dispensa uma URL externa.
`scripts/stage_deploy_assets.py` copia o bundle, `result.json` e
`test_predictions.npz` de `manometry-classification` para `assets/model/`.

**Torch de CPU.** `requirements-cpu.txt` fixa a wheel sem CUDA. A padrão traz o
runtime da NVIDIA e passa de 2 GB, o que estoura o build.

**Latência.** Uma análise de dez deglutições leva **cerca de 1,8 s** em duas
threads de CPU (notebook) depois da primeira requisição, que paga o aquecimento;
o encoder responde por 1,5 s disso (8,9 GFLOPs por deglutição) e o bundle sobe em
cerca de 3 s. As miniaturas PNG da resposta somam outros 2 a 3 s. Numa GPU o
forward cai para 3 ms. `MANOMETRY_TORCH_THREADS=2` costuma ser melhor que deixar o
torch abrir uma thread por core anunciado pela máquina, dos quais o container só
recebe uma fração.

**Primeira requisição.** Nos planos que hibernam, ela paga o boot inteiro. Com
`MANOMETRY_LAZY_MODEL=false` o checkpoint entra na memória antes de o health
check passar, e o Render só encaminha tráfego depois disso.

**CORS.** `MANOMETRY_ALLOWED_ORIGINS` com o domínio do Vercel e
`MANOMETRY_ALLOWED_ORIGIN_REGEX=https://.*\.vercel\.app` para os previews.

**Imagens de referência e laudos de exemplo** precisam do corpus, que não existe
numa instância hospedada. `scripts/export_reference_images.py` e
`scripts/build_sample_reports.py` geram os arquivos localmente em
`assets/reference/` e `assets/samples/`, ambos ignorados pelo git. Publicá-los é
uma decisão sobre expor imagens derivadas de pacientes, e é de quem faz o
deploy. Sem eles, o atlas cai para os esquemas em SVG e o botão de laudo de
exemplo não aparece.

## Scripts

```bash
scripts/sync_vendor.sh                        # re-copia os módulos de origem
python3 scripts/export_reference_images.py    # topografia de referência de cada classe
python3 scripts/build_sample_reports.py       # laudos de demonstração
python3 scripts/stage_deploy_assets.py        # artefatos pequenos para o deploy
python3 scripts/make_sample_docx.py --class achalasia_type_ii --out demo.docx
```

## Testes

```bash
python3 -m pytest              # rápido: leitor, recorte, texto clínico, HTTP
python3 -m pytest --runslow    # inclui a paridade contra o treino
```

## Limites que este serviço não esconde

- **As métricas são estimativas a partir da imagem.** DCI, IRP, latência e os
  estados por deglutição saem das cabeças de medida do modelo; quebras e JEG
  basal, da inversão da rampa de cor do gráfico. Servem para justificar a
  decisão; não são medidas do sistema de aquisição e não devem ser transcritas
  para um laudo.
- **Duas classes da v4.0 estão fora**: espasmo esofágico distal (n=4) e acalásia
  tipo III (n=3), depois da recontagem das deglutições prematuras com o corte de
  4,5 s da v4.0. Um exame dessas condições será forçado à classe mais próxima.
- **Obstrução da saída da JEG é a classe fraca** (18 exames no corpus, F1 0,50
  nos 5 de teste do fold) e segue uma regra simplificada: IRP elevada sem 100% de
  falhas. A ressalva viaja junto do texto clínico da classe.
- **Um fold, não o ensemble do artigo.** Os números do artigo são a média de 75
  folds; o serviço carrega um deles. O desempenho dele está acima da média, mas
  num teste de 358 exames.
- **As imagens do corpus trazem as anotações de medida do próprio software**:
  caixas, distâncias em cm, tempos em s e valores de DCI desenhados sobre o
  traçado. Elas estão presentes tanto no treino quanto no teste, então parte do
  desempenho relatado pode vir de o codificador aprender a lê-las em vez de ler
  a topografia. Remover as anotações e repetir a medição é trabalho para o
  repositório de classificação.
- **Contexto clínico não entra.** A v4.0 exige sintomas para nomear esôfago
  hipercontrátil e espasmo distal como distúrbio, e uma imagem não os carrega.

Ferramenta de pesquisa e apoio à decisão. Não substitui a leitura do exame por
um médico habilitado.
