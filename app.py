"""Protótipo da aplicação (seção 3.1.3) - Streamlit.

Executar localmente:  streamlit run app/app.py
"""
import streamlit as st
from PIL import Image

import inferencia

st.set_page_config(page_title="Mosaico do feijoeiro", page_icon="🌱", layout="centered")


@st.cache_resource(show_spinner="Carregando o modelo...")
def obter_modelo():
    return inferencia.carregar()


st.title("🌱 Triagem de mosaico em folhas de feijoeiro")
st.write("Fotografe uma folha de feijoeiro ou escolha uma imagem da galeria. "
         "Enquadre **uma folha**, bem iluminada e em foco.")

if not (inferencia.PASTA_MODELO / "modelo_bgmv.keras").exists():
    st.error("Modelo não encontrado em app/modelo/. Execute o treinamento (codigo/04_treinar.py).")
    st.stop()
modelo, info = obter_modelo()

aba_camera, aba_galeria = st.tabs(["📷 Câmera", "🖼️ Galeria"])
with aba_camera:
    foto = st.camera_input("Fotografar a folha")
with aba_galeria:
    arquivo = st.file_uploader("Escolher imagem", type=["jpg", "jpeg", "png", "webp"])

entrada = foto or arquivo
if entrada:
    try:
        imagem = Image.open(entrada)
        resultado = inferencia.classificar(modelo, info, imagem)
    except Exception:
        st.error("Não foi possível ler a imagem. Tente outra fotografia.")
        st.stop()

    st.image(imagem, caption="Imagem analisada")
    if resultado["confiavel"]:
        if resultado["classe"] == "saudavel":
            st.success(f"**Diagnóstico provável:** {resultado['nome']}")
        else:
            st.warning(f"**Diagnóstico provável:** {resultado['nome']}")
        st.metric("Confiança", f"{resultado['confianca']:.0%}")
        st.write(resultado["orientacao"])
    else:
        st.error(f"**Resultado inconclusivo** (confiança de {resultado['confianca']:.0%}, "
                 f"abaixo de {info.get('confianca_minima', 0.70):.0%}).")
        st.write("Tire uma **nova fotografia**, mais próxima, com boa luz e apenas uma folha no "
                 "quadro. Se o resultado continuar inconclusivo, **consulte um técnico**.")
    with st.expander("Detalhes da classificação"):
        for nome, p in sorted(resultado["probabilidades"].items(), key=lambda x: -x[1]):
            st.progress(p, text=f"{nome}: {p:.0%}")

st.divider()
st.caption("Ferramenta de triagem e apoio à decisão, desenvolvida em um Trabalho de Conclusão de "
           "Curso. Foi treinada com imagens de bases públicas, que não distinguem o mosaico dourado "
           "de outras viroses do tipo mosaico, e seu desempenho em lavouras não foi avaliado. "
           "Não substitui o diagnóstico de um profissional habilitado.")
