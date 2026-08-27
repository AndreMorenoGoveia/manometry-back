# manometry-back

API que recebe o laudo de uma manometria esofágica de alta resolução, recorta as
topografias de pressão das dez deglutições de água e devolve a classificação do
exame pela **Classificação de Chicago v4.0**, junto das métricas que sustentam a
decisão.

O modelo vem de [`manometry-classification`](../manometry-classification):
codificador ConvNeXt-Base pré-treinado em ImageNet-22k sobre cada deglutição e
agregação `chicago_set` no nível do exame, que combina um ramo com as
posteriores por deglutição e as métricas de Chicago a um Set Transformer sobre
os embeddings. Partição estrita por paciente, oito classes.

```
laudo .docx/.rtf
   │  leitura dos PNG na ordem do documento
   ▼
paineis  ──▶ retrato = deglutição · paisagem = logotipo
   │
   │  unifier.images.clean_swallow  (código do construtor do dataset, verbatim)
   ▼
topografia recortada ──┬──▶ inversão da rampa jet ──▶ 24×241 ──▶ DCI, IRP4, DL, quebras
                       │                                              │
                       └──▶ 224px ──▶ codificador ──▶ embeddings ─────┤
                                                                      ▼
                                            ramo Chicago + Set Transformer ──▶ 8 classes
```

## Qual modelo o serviço carrega

O checkpoint anteriormente marcado como *deploy* no repositório de classificação
foi treinado com as métricas de Chicago derivadas dos **CSV de pressão**
(`.cache/institutional_metrics.npy`). Um laudo não traz esses CSV, só imagens,
então servi-lo aqui alimentaria o ramo Chicago com uma distribuição que ele nunca
viu. O commit seguinte daquele repositório introduziu
`experiments/image_chicago.py`, que recupera as mesmas métricas exclusivamente da
imagem renderizada, mas nenhum modelo chegou a ser treinado nessa configuração.

Foram treinados dois, com a mesma receita e a mesma partição estrita por
paciente, mudando só o codificador:

| run | codificador | checkpoint | multi-crop | val macro-F1 | test macro-F1 | test acurácia |
|---|---|---:|:--:|---:|---:|---:|
| `fusion_light_deploy` (padrão) | ResNet-18 | 50 MB | não | **0,771** | 0,738 | **0,780** |
| `fusion_heavy_image_deploy` | ConvNeXt-Base 22k | 342 MB | sim | 0,713 | 0,741 | 0,750 |

O padrão é o leve: empata em macro-F1 e ganha em acurácia usando um sétimo dos
pesos e um quinto do cálculo, o que é a diferença entre caber e não caber num
container sem GPU. Uma ressalva de honestidade: o run leve varreu três sementes
e o pesado apenas uma, então a comparação direta favorece o leve pelo orçamento
de busca. O que se sustenta é que ele não perde nada e custa muito menos.

`scripts/compare_checkpoints.py` ordena os candidatos pela macro-F1 de validação,
que é o único número pelo qual um checkpoint pode ser escolhido, e recusa os que
padronizaram as features a partir dos CSV: ele refaz a partição de treino do run
e compara a média recomputada das métricas de imagem com a constante guardada no
checkpoint.

O treino grava também se ficou com o recorte central ou com a média de cinco
recortes, e o serviço repete essa escolha. `MANOMETRY_TTA` inverte a decisão,
porque cinco passadas podem não valer a diferença num host sem GPU.

`tests/test_inference_parity.py` reexecuta a partição de teste do treino pelo
caminho da API e exige que as probabilidades batam com as guardadas junto ao
checkpoint, com desvio médio abaixo de 1e-3. É o que impede a transcrição da
arquitetura de sair do lugar. `tests/test_end_to_end.py` fecha o circuito pelo
HTTP, com laudos montados a partir de exames reais.

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
| `MANOMETRY_CHECKPOINT` | `<root>/artifacts/leakage_fusion/fusion_light_deploy/.../seed_43/model.pt` | outro checkpoint |
| `MANOMETRY_CHECKPOINT_URL` | vazio | baixa o checkpoint no primeiro boot, se o arquivo local não existir |
| `MANOMETRY_RESULT_JSON` | `result.json` ao lado do checkpoint | métricas publicadas em `/api/model` |
| `MANOMETRY_PREDICTIONS` | `test_predictions.npz` ao lado do checkpoint | desempenho por classe |
| `MANOMETRY_CALIBRATION` | `<root>/.cache/image_chicago_calibration.json` | calibração da rampa de cor |
| `MANOMETRY_DATASET_ROOT` | `<root>/data` | corpus, para imagens de referência e laudos de exemplo |
| `MANOMETRY_REFERENCE_DIR` | `assets/reference` | PNG de referência já exportados |
| `MANOMETRY_SAMPLE_DIR` | `assets/samples` | laudos de demonstração já gerados |
| `MANOMETRY_DEVICE` | `auto` | `cuda`, `cpu` |
| `MANOMETRY_TTA` | segue o treino | `false` usa só o recorte central, 5x menos cálculo |
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
| `GET` | `/api/diseases` | texto clínico das oito classes |
| `GET` | `/api/diseases/{slug}/reference.png` | topografia real da classe |
| `GET` | `/api/vocabulary` | rótulos dos padrões de deglutição e das métricas |
| `GET` | `/api/samples` | classes com laudo de exemplo nesta instância |
| `GET` | `/api/sample-report?diagnosis=<slug>` | laudo `.docx` de demonstração |

O arquivo enviado nunca toca o disco: é lido em memória, as imagens são
recortadas, medidas e descartadas com a resposta.

## Deploy no Render

`render.yaml` já traz o blueprint. O que decide se funciona:

**Memória.** Servindo o modelo leve o processo chega perto de **1 GB** de pico,
somando o runtime do torch, os pesos e as ativações de dez imagens; com o
ConvNeXt-Base, some mais 350 MB só de pesos. Em ambos os casos o plano é o
**Standard (2 GB)**; os de 512 MB não sustentam nem o leve.

**O arquivo de pesos.** São 50 MB no modelo leve e 342 MB no pesado. Publique-o
como release asset e aponte `MANOMETRY_CHECKPOINT_URL` para a URL: o primeiro
boot baixa para `MANOMETRY_CHECKPOINT` e os seguintes reaproveitam enquanto o
container viver. (Os 50 MB também caberiam versionados, abaixo do limite de 100
MB do GitHub, se preferir zero configuração.) `scripts/stage_deploy_assets.py`
copia para `assets/model/` os três arquivos pequenos que acompanham o checkpoint
(calibração, `result.json` e `test_predictions.npz`), que são versionáveis.

**Torch de CPU.** `requirements-cpu.txt` fixa a wheel sem CUDA. A padrão traz o
runtime da NVIDIA e passa de 2 GB, o que estoura o build.

**Latência.** Com o modelo leve, uma análise de dez deglutições leva **cerca de
0,7 s** em duas threads de CPU depois da primeira requisição, que paga o
aquecimento; o checkpoint sobe em pouco mais de 4 s. Com o ConvNeXt-Base e
multi-crop, a mesma análise passa de um minuto. `MANOMETRY_TORCH_THREADS=2`
costuma ser melhor que deixar o torch abrir uma thread por core anunciado pela
máquina, dos quais o container só recebe uma fração, e `MANOMETRY_TTA=false`
corta o cálculo por cinco quando o checkpoint escolhido usa multi-crop.

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

- **As métricas são proxies de imagem.** DCI, IRP, latência e quebras são
  recuperadas invertendo a rampa de cor do gráfico, ancorada numa calibração
  global ajustada apenas em exames normais de treino. Servem para justificar a
  decisão; não são medidas do sistema de aquisição e não devem ser transcritas
  para um laudo.
- **Duas classes da v4.0 estão fora**: obstrução da saída da JEG (n=12) e
  acalásia tipo III (n=5) não têm exames suficientes para uma partição em três
  partes. Um exame dessas condições será forçado à classe mais próxima.
- **`hypotensive_egj` não é um diagnóstico da v4.0**, e sim o rótulo que o
  serviço de origem usa para barreira antirrefluxo incompetente com corpo
  esofágico normal. A ressalva viaja junto do texto clínico da classe.
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
