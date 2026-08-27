"""Clinical reference for the eight diagnoses the classifier can return.

Every entry follows the Chicago Classification v4.0 (Yadlapati et al., *Neurogastroenterol
Motil* 2021;33:e14058), which is the framework the reports in the training corpus
were written against.  Two entries need a word of warning and carry it in their own
text: `hypotensive_egj` is not a CC v4.0 motility diagnosis but a barrier finding
this service's corpus labels explicitly, and `hypercontractile_esophagus` is a
manometric pattern that CC v4.0 only calls a disorder in the presence of dysphagia
or non-cardiac chest pain, and manometry alone cannot close that diagnosis.

The numeric cut-offs quoted here are the CC v4.0 ones.  The values this service
computes from an uploaded report are *image-calibrated proxies* recovered from the
rendered topography, so they are reported next to the cut-offs as supporting
evidence for the reader, never as measurements made by the acquisition system.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

FAMILY_EGJ = "Obstrução da saída da junção esofagogástrica"
FAMILY_PERISTALSIS = "Distúrbios da peristalse"
FAMILY_BARRIER = "Achado de barreira antirrefluxo"
FAMILY_NORMAL = "Exame normal"


@dataclass(frozen=True)
class Disease:
    slug: str
    name: str
    short_name: str
    english: str
    family: str
    accent: str
    tagline: str
    summary: str
    criteria: list[str]
    topography: str
    symptoms: list[str]
    management: str
    differential: list[str]
    metric_signature: dict[str, str]
    prevalence: int
    caveat: str = ""
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_CC40 = "Chicago Classification v4.0. Yadlapati R, et al. Neurogastroenterol Motil. 2021;33:e14058"
_HRM_PRIMER = "Gyawali CP, et al. Esophageal motility disorders: HRM interpretation. Am J Gastroenterol. 2020"

DISEASES: tuple[Disease, ...] = (
    Disease(
        slug="normal_motility",
        name="Motilidade esofágica normal",
        short_name="Normal",
        english="Normal esophageal motility",
        family=FAMILY_NORMAL,
        accent="#34D399",
        tagline="Relaxamento normal da JEG e peristalse efetiva",
        summary=(
            "O exame não preenche critério para nenhum distúrbio motor da Classificação de Chicago v4.0. "
            "A junção esofagogástrica relaxa normalmente a cada deglutição e a onda peristáltica percorre "
            "o corpo esofágico com vigor e latência preservados."
        ),
        criteria=[
            "IRP mediana dentro do limite superior da normalidade (≤ 15 mmHg em cateter de estado sólido).",
            "Mais de 50% das deglutições efetivas (DCI ≥ 450 mmHg·s·cm).",
            "Menos de 20% de deglutições prematuras e menos de 20% hipercontráteis.",
            "Ausência de pressurização panesofágica significativa.",
        ],
        topography=(
            "Faixa de contração distal contínua e bem definida, cruzando o contorno isobárico de 20 mmHg "
            "sem quebras maiores que 5 cm, com a JEG apagando-se (relaxando) logo após a deglutição."
        ),
        symptoms=[
            "Frequentemente assintomático do ponto de vista motor",
            "Sintomas de refluxo, quando presentes, tendem a ter outra explicação (barreira, sensibilidade)",
        ],
        management=(
            "Nenhum tratamento motor é indicado. Quando o paciente foi encaminhado por disfagia ou dor "
            "torácica, a investigação segue por outras vias: pHmetria/impedância, endoscopia, avaliação "
            "de hipersensibilidade visceral ou causas não esofágicas."
        ),
        differential=[
            "Motilidade esofágica ineficaz limítrofe",
            "Distúrbio motor intermitente não capturado nas dez deglutições de água",
        ],
        metric_signature={
            "irp4": "≤ 15 mmHg",
            "dci": "450 – 8.000 mmHg·s·cm na maioria das deglutições",
            "distal_latency": "≥ 4,5 s",
        },
        prevalence=470,
        references=[_CC40, _HRM_PRIMER],
    ),
    Disease(
        slug="ineffective_esophageal_motility",
        name="Motilidade esofágica ineficaz",
        short_name="MEI",
        english="Ineffective esophageal motility (IEM)",
        family=FAMILY_PERISTALSIS,
        accent="#FBBF24",
        tagline="Peristalse fraca ou falha, com JEG que relaxa normalmente",
        summary=(
            "Distúrbio menor da peristalse: a JEG relaxa bem, mas o corpo esofágico contrai de forma fraca "
            "ou falha em boa parte das deglutições. É o achado motor mais comum em pacientes com refluxo e "
            "o principal responsável por clareamento esofágico lento."
        ),
        criteria=[
            "IRP mediana normal (≤ 15 mmHg): a JEG não obstrui.",
            "Mais de 70% das deglutições ineficazes (DCI < 450 mmHg·s·cm), ou",
            "Ao menos 50% de deglutições falhas (DCI < 100 mmHg·s·cm).",
            "A v4.0 elevou o limiar em relação à v3.0 justamente para reduzir sobrediagnóstico.",
        ],
        topography=(
            "Faixa de contração distal fragmentada ou apagada: grandes quebras no contorno de 20 mmHg, "
            "amplitude baixa e, nas deglutições falhas, ausência quase completa de sinal contrátil no corpo, "
            "com a JEG ainda relaxando normalmente."
        ),
        symptoms=[
            "Pirose e regurgitação (associação forte com DRGE)",
            "Disfagia leve e intermitente, sobretudo para sólidos",
            "Sensação de alimento parado que melhora com líquidos",
        ],
        management=(
            "Não há tratamento motor específico eficaz. A conduta é dirigida ao refluxo (inibidor de bomba "
            "de prótons, medidas posturais e de peso) e à orientação alimentar (mastigação, líquidos junto "
            "às refeições). Em avaliação pré-fundoplicatura o achado é relevante para calibrar a válvula."
        ),
        differential=[
            "Contratilidade ausente (100% de deglutições falhas)",
            "Motilidade normal com deglutições ineficazes isoladas",
            "Esclerose sistêmica e outras doenças do tecido conjuntivo",
        ],
        metric_signature={
            "irp4": "≤ 15 mmHg",
            "dci": "< 450 mmHg·s·cm em mais de 70% das deglutições",
            "distal_latency": "normal quando há contração mensurável",
        },
        prevalence=354,
        references=[_CC40, _HRM_PRIMER],
    ),
    Disease(
        slug="hypotensive_egj",
        name="Junção esofagogástrica hipotensiva",
        short_name="JEG hipotensiva",
        english="Hypotensive esophagogastric junction / hiatal hernia",
        family=FAMILY_BARRIER,
        accent="#38BDF8",
        tagline="Barreira antirrefluxo incompetente com corpo esofágico preservado",
        summary=(
            "Não é um distúrbio de motilidade: é a descrição manométrica de uma barreira antirrefluxo fraca, "
            "com pressão basal da JEG baixa e frequentemente com separação entre o esfíncter inferior e a "
            "crura diafragmática (hérnia hiatal), em um esôfago cujo corpo contrai normalmente."
        ),
        criteria=[
            "IRP mediana normal: não há obstrução da saída.",
            "Peristalse do corpo esofágico preservada (não preenche critério de MEI nem de espasmo).",
            "Pressão basal da JEG reduzida e/ou morfologia tipo II–III (separação JEG–crura).",
        ],
        topography=(
            "Peristalse distal normal, mas banda de alta pressão da JEG tênue no traçado basal e, na hérnia "
            "hiatal, duas bandas separadas (o esfíncter inferior acima e a crura diafragmática abaixo) em "
            "vez de uma única zona de alta pressão."
        ),
        symptoms=[
            "Pirose e regurgitação, tipicamente pós-prandiais e posturais",
            "Refluxo noturno",
            "Sintomas extraesofágicos: tosse, rouquidão, pigarro",
        ],
        management=(
            "A conduta é a da doença do refluxo: inibidor de bomba de prótons, perda de peso, elevação da "
            "cabeceira. A caracterização da JEG e do tamanho da hérnia é parte do preparo cirúrgico quando "
            "se cogita fundoplicatura ou reparo do hiato."
        ),
        differential=[
            "Motilidade normal com JEG de pressão limítrofe",
            "Motilidade esofágica ineficaz associada a refluxo",
        ],
        metric_signature={
            "irp4": "≤ 15 mmHg",
            "dci": "predominantemente ≥ 450 mmHg·s·cm",
            "egj_basal": "pressão basal da JEG reduzida",
        },
        prevalence=327,
        caveat=(
            "A Classificação de Chicago v4.0 não lista “JEG hipotensiva” entre os diagnósticos motores: "
            "ela descreve a morfologia da JEG e a barreira antirrefluxo em separado. Esta classe reproduz "
            "o rótulo usado nos laudos do serviço de origem, onde descreve exames com barreira incompetente "
            "e corpo esofágico normal."
        ),
        references=[
            _CC40,
            "Gyawali CP, et al. Modern diagnosis of GERD: the Lyon Consensus. Gut. 2018;67:1351-1362",
        ],
    ),
    Disease(
        slug="absent_contractility",
        name="Contratilidade ausente",
        short_name="Contratilidade ausente",
        english="Absent contractility",
        family=FAMILY_PERISTALSIS,
        accent="#94A3B8",
        tagline="Aperistalse completa com relaxamento normal da JEG",
        summary=(
            "Todas as dez deglutições falham: não há qualquer contração organizada no corpo esofágico. "
            "A diferença crucial em relação à acalásia é que a JEG relaxa normalmente: o esôfago não empurra, "
            "mas também não há obstrução na saída."
        ),
        criteria=[
            "IRP mediana normal (≤ 15 mmHg).",
            "100% das deglutições falhas (DCI < 100 mmHg·s·cm).",
            "Se a IRP estiver limítrofe, a v4.0 orienta reconsiderar acalásia e complementar com testes "
            "provocativos (deglutições múltiplas rápidas, teste de deglutição rápida, esofagograma temporizado).",
        ],
        topography=(
            "Traçado essencialmente plano ao longo de todo o corpo esofágico após a deglutição, sem faixa "
            "de contração distal, e com a JEG apagando-se normalmente na janela deglutitória."
        ),
        symptoms=[
            "Disfagia para sólidos e líquidos",
            "Refluxo grave, muitas vezes com esofagite péptica de difícil controle",
            "Regurgitação e clareamento esofágico muito lento",
        ],
        management=(
            "A busca de causa sistêmica é obrigatória: esclerose sistêmica é a associação clássica, e a "
            "aperistalse com JEG hipotônica é praticamente a assinatura manométrica da doença. O tratamento "
            "é antirrefluxo agressivo e orientação alimentar; a fundoplicatura total é evitada pelo risco de "
            "disfagia em um esôfago que não empurra."
        ),
        differential=[
            "Acalásia tipo I, em que a IRP está elevada",
            "Motilidade esofágica ineficaz grave, em que ainda restam deglutições efetivas",
            "Artefato de aquisição / exame incompleto",
        ],
        metric_signature={
            "irp4": "≤ 15 mmHg",
            "dci": "< 100 mmHg·s·cm em 100% das deglutições",
            "distal_latency": "não mensurável, pois não há contração",
        },
        prevalence=109,
        references=[
            _CC40,
            "Roman S, Kahrilas PJ. Management of spastic disorders of the esophagus. Gastroenterol Clin North Am. 2013",
        ],
    ),
    Disease(
        slug="achalasia_type_i",
        name="Acalásia tipo I (clássica)",
        short_name="Acalásia I",
        english="Type I achalasia (classic)",
        family=FAMILY_EGJ,
        accent="#A78BFA",
        tagline="JEG que não relaxa somada a aperistalse, sem pressurização",
        summary=(
            "Forma clássica: a junção esofagogástrica não relaxa e o corpo esofágico está completamente "
            "aperistáltico, sem gerar pressão. É o subtipo do esôfago já dilatado e descompensado, que perdeu "
            "a capacidade de pressurizar contra a obstrução."
        ),
        criteria=[
            "IRP mediana elevada (> 15 mmHg em cateter de estado sólido).",
            "100% de peristalse ausente.",
            "Pressurização panesofágica em menos de 20% das deglutições.",
        ],
        topography=(
            "Coluna esofágica silenciosa, sem faixa de contração e sem pressurização, sob uma banda de alta "
            "pressão persistente na JEG que não se apaga durante a deglutição."
        ),
        symptoms=[
            "Disfagia progressiva para sólidos e líquidos",
            "Regurgitação de alimento não digerido, sobretudo noturna",
            "Perda de peso",
            "Dor torácica menos frequente que nos tipos II e III",
        ],
        management=(
            "Tratamento dirigido à JEG: miotomia endoscópica peroral (POEM), miotomia de Heller "
            "laparoscópica com fundoplicatura parcial ou dilatação pneumática progressiva. A resposta ao "
            "tratamento é, em média, menor que a do tipo II, refletindo o esôfago mais avançado."
        ),
        differential=[
            "Acalásia tipo II, com pressurização panesofágica",
            "Contratilidade ausente, com IRP normal",
            "Pseudoacalásia (neoplasia da cárdia), que obriga endoscopia e, quando suspeita, TC ou eco-endoscopia",
        ],
        metric_signature={
            "irp4": "> 15 mmHg",
            "dci": "< 100 mmHg·s·cm em todas as deglutições",
            "pressurization": "< 20% das deglutições",
        },
        prevalence=161,
        references=[
            _CC40,
            "Pandolfino JE, et al. Achalasia: a new clinically relevant classification by HRM. Gastroenterology. 2008;135:1526-33",
        ],
    ),
    Disease(
        slug="achalasia_type_ii",
        name="Acalásia tipo II (com pressurização panesofágica)",
        short_name="Acalásia II",
        english="Type II achalasia (with pan-esophageal pressurization)",
        family=FAMILY_EGJ,
        accent="#E879F9",
        tagline="Obstrução da JEG com compressão panesofágica do bolo",
        summary=(
            "A JEG não relaxa e o corpo esofágico, ainda que aperistáltico, comprime o bolo em toda a sua "
            "extensão ao mesmo tempo, gerando pressurização uniforme entre o esfíncter superior e a JEG. "
            "É o subtipo mais comum e o de melhor resposta ao tratamento."
        ),
        criteria=[
            "IRP mediana elevada (> 15 mmHg).",
            "100% de peristalse ausente.",
            "Pressurização panesofágica em ao menos 20% das deglutições.",
        ],
        topography=(
            "Faixa horizontal de pressão homogênea ocupando todo o corpo esofágico simultaneamente, sem a "
            "inclinação de uma onda peristáltica, retida acima de uma JEG que permanece fechada."
        ),
        symptoms=[
            "Disfagia para sólidos e líquidos",
            "Dor torácica e sensação de aperto retroesternal, mais frequentes que no tipo I",
            "Regurgitação",
            "Perda de peso",
        ],
        management=(
            "Mesmo arsenal do tipo I (POEM, miotomia de Heller com fundoplicatura parcial ou dilatação "
            "pneumática), porém com as melhores taxas de sucesso entre os subtipos, acima de 90% em séries "
            "cirúrgicas. O esôfago ainda gera pressão, o que sinaliza doença menos avançada."
        ),
        differential=[
            "Acalásia tipo I, sem pressurização panesofágica",
            "Obstrução da saída da JEG (EGJOO), com peristalse preservada em alguma medida",
            "Pseudoacalásia, que obriga endoscopia",
        ],
        metric_signature={
            "irp4": "> 15 mmHg",
            "dci": "< 100 mmHg·s·cm (sem contração verdadeira)",
            "pressurization": "≥ 20% das deglutições com pressurização panesofágica",
        },
        prevalence=168,
        references=[
            _CC40,
            "Rohof WO, et al. Outcomes of treatment for achalasia depend on manometric subtype. Gastroenterology. 2013;144:718-25",
        ],
    ),
    Disease(
        slug="distal_esophageal_spasm",
        name="Espasmo esofágico distal",
        short_name="Espasmo distal",
        english="Distal esophageal spasm (DES)",
        family=FAMILY_PERISTALSIS,
        accent="#FB923C",
        tagline="Contrações prematuras: o esôfago contrai antes da hora",
        summary=(
            "Distúrbio maior da peristalse em que a contração perde o sequenciamento e chega ao esôfago "
            "distal cedo demais. O marcador é a latência distal curta, não a amplitude: o problema é "
            "o momento da contração, não a sua força."
        ),
        criteria=[
            "IRP mediana normal (≤ 15 mmHg). Se estiver elevada, o diagnóstico passa a ser acalásia tipo III.",
            "Ao menos 20% de deglutições prematuras: latência distal < 4,5 s com DCI > 450 mmHg·s·cm.",
            "A v4.0 exige sintomas clinicamente relevantes (disfagia ou dor torácica não cardíaca).",
        ],
        topography=(
            "A faixa de contração perde a inclinação diagonal característica e aparece quase vertical ou "
            "simultânea no esôfago distal, alcançando o ponto de desaceleração contrátil antes de 4,5 s do "
            "início da deglutição."
        ),
        symptoms=[
            "Dor torácica não cardíaca, frequentemente intensa e em aperto",
            "Disfagia intermitente para sólidos e líquidos",
            "Sintomas desencadeados por líquidos muito frios ou quentes e por estresse",
        ],
        management=(
            "Investigação obrigatória de causa cardíaca antes de atribuir a dor ao esôfago. O tratamento é "
            "sintomático: relaxantes de músculo liso (nitratos, bloqueadores de canal de cálcio), inibidores "
            "de fosfodiesterase, neuromoduladores em dose baixa e tratamento de refluxo associado; POEM fica "
            "reservado a casos refratários e bem caracterizados."
        ),
        differential=[
            "Acalásia tipo III, com a mesma prematuridade porém IRP elevada",
            "Esôfago hipercontrátil, com DCI muito alto mas latência normal",
            "Espasmo secundário a refluxo ou a obstrução mecânica",
        ],
        metric_signature={
            "irp4": "≤ 15 mmHg",
            "dci": "> 450 mmHg·s·cm nas deglutições prematuras",
            "distal_latency": "< 4,5 s em ao menos 20% das deglutições",
        },
        prevalence=54,
        references=[
            _CC40,
            "Roman S, Kahrilas PJ. Distal esophageal spasm. Curr Opin Gastroenterol. 2015;31:328-33",
        ],
    ),
    Disease(
        slug="hypercontractile_esophagus",
        name="Esôfago hipercontrátil (jackhammer)",
        short_name="Hipercontrátil",
        english="Hypercontractile esophagus (jackhammer)",
        family=FAMILY_PERISTALSIS,
        accent="#F87171",
        tagline="Contrações de vigor extremo com sequência preservada",
        summary=(
            "A peristalse mantém a sequência e a latência normais, mas o vigor da contração é excessivo: "
            "o integral contrátil distal ultrapassa 8.000 mmHg·s·cm, muitas vezes com contrações repetitivas "
            "sobrepostas que dão ao traçado o aspecto de “martelete”."
        ),
        criteria=[
            "IRP mediana normal (≤ 15 mmHg).",
            "Ao menos 20% de deglutições com DCI > 8.000 mmHg·s·cm.",
            "Latência distal normal (≥ 4,5 s). Do contrário trata-se de espasmo ou acalásia tipo III.",
            "Sintomas clinicamente relevantes são exigidos pela v4.0 para nomear a condição como distúrbio.",
        ],
        topography=(
            "Faixa de contração distal muito intensa e alargada, com picos de pressão elevados e frequentemente "
            "múltiplos vigorosos em uma mesma deglutição, mantendo a inclinação normal da onda peristáltica."
        ),
        symptoms=[
            "Dor torácica não cardíaca",
            "Disfagia",
            "Sintomas de refluxo concomitantes são comuns",
        ],
        management=(
            "Excluir causa cardíaca e obstrução distal, inclusive esofagite eosinofílica, que pode cursar com "
            "hipercontratilidade secundária. O tratamento começa por refluxo e neuromoduladores; relaxantes de "
            "músculo liso e, em casos refratários e bem selecionados, POEM."
        ),
        differential=[
            "Espasmo esofágico distal, com latência curta",
            "Hipercontratilidade secundária a obstrução distal ou esofagite eosinofílica",
            "Acalásia tipo III",
        ],
        metric_signature={
            "irp4": "≤ 15 mmHg",
            "dci": "> 8.000 mmHg·s·cm em ao menos 20% das deglutições",
            "distal_latency": "≥ 4,5 s",
        },
        prevalence=38,
        caveat=(
            "A v4.0 considera o padrão “jackhammer” um distúrbio apenas quando acompanhado de disfagia ou dor "
            "torácica não cardíaca. O achado manométrico isolado, em paciente assintomático, não fecha "
            "diagnóstico, e deve levar à busca de uma obstrução distal que explique a hipercontratilidade."
        ),
        references=[
            _CC40,
            "Kahrilas PJ, et al. Expert consensus on the use of HRM in clinical practice. Neurogastroenterol Motil. 2017",
        ],
    ),
)

BY_SLUG: dict[str, Disease] = {disease.slug: disease for disease in DISEASES}

# The order the network's output vector is in: alphabetical by slug, as the
# training corpus enumerated the class directories.
CLASS_ORDER: tuple[str, ...] = tuple(sorted(BY_SLUG))


def get(slug: str) -> Disease:
    try:
        return BY_SLUG[slug]
    except KeyError as error:
        raise KeyError(f"Diagnostico desconhecido: {slug}") from error
