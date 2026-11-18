"""Inferência do protótipo: carrega o modelo e classifica uma imagem.

Usado pelo aplicativo (app.py) e pelo teste do protótipo
(codigo/07_testar_prototipo.py), garantindo que ambos executem o mesmo código.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PASTA_MODELO = Path(__file__).resolve().parent / "modelo"

ORIENTACOES = {
    "mosaico_dourado": (
        "Sintomas compatíveis com mosaico viral. Procure a assistência técnica (agrônomo ou "
        "extensão rural) para confirmar o diagnóstico. O vírus do mosaico dourado é transmitido "
        "pela mosca-branca: monitore e controle o inseto conforme orientação técnica e evite "
        "novos plantios ao lado de lavouras infectadas."),
    "saudavel": (
        "Não foram identificados sintomas nesta folha. Continue o monitoramento periódico da "
        "lavoura, em especial a presença de mosca-branca."),
    "outras_alteracoes": (
        "A folha apresenta alterações que não parecem ser de mosaico (por exemplo, ferrugem ou "
        "mancha-angular). Procure a assistência técnica para identificar a causa e o manejo adequado."),
}


def carregar():
    """Devolve (modelo, info). info traz classes, nomes de exibição e confiança mínima."""
    import keras
    info = json.loads((PASTA_MODELO / "info_modelo.json").read_text(encoding="utf-8"))
    modelo = keras.models.load_model(PASTA_MODELO / "modelo_bgmv.keras", compile=False)
    return modelo, info


def preparar(imagem: Image.Image, tamanho: int = 224) -> np.ndarray:
    """Mesmo pré-processamento do treinamento: orientação EXIF, RGB, 224 x 224."""
    imagem = ImageOps.exif_transpose(imagem).convert("RGB")
    imagem = imagem.resize((tamanho, tamanho), Image.BILINEAR)
    return np.asarray(imagem, dtype="float32")[None]


def classificar(modelo, info, imagem: Image.Image) -> dict:
    prob = modelo.predict(preparar(imagem, info.get("tamanho", 224)), verbose=0)[0]
    i = int(prob.argmax())
    classe = info["classes"][i]
    confianca = float(prob[i])
    return dict(
        classe=classe,
        nome=info["nomes_exibicao"].get(classe, classe),
        confianca=confianca,
        confiavel=confianca >= info.get("confianca_minima", 0.70),
        orientacao=ORIENTACOES.get(classe, ""),
        probabilidades={info["nomes_exibicao"].get(c, c): float(p) for c, p in zip(info["classes"], prob)},
    )
