# Triagem de mosaico em folhas de feijoeiro

Protótipo do Trabalho de Conclusão de Curso: aplicativo web que classifica a foto de
uma folha de feijoeiro em **mosaico viral** (sintomas compatíveis com mosaico dourado),
**folha saudável** ou **outras alterações**, usando uma rede MobileNetV2 treinada por
transferência de aprendizado.

## Usar
Abra o link do aplicativo no celular ou no computador, fotografe uma folha (ou escolha
uma imagem da galeria) e veja o diagnóstico provável, a confiança e a orientação.
Resultados com confiança abaixo de 70% aparecem como inconclusivos.

## Rodar no próprio computador
```
pip install -r requirements.txt
streamlit run app.py
```

## Arquivos
- `app.py` – interface (Streamlit)
- `inferencia.py` – carregamento do modelo e classificação
- `modelo/modelo_bgmv.keras` – modelo treinado; `modelo/info_modelo.json` – classes e parâmetros

## Limitações
Modelo treinado com imagens de uma base pública (folhas fotografadas sobre fundo claro),
que não distingue o mosaico dourado (BGMV) de outras viroses do tipo mosaico. O
desempenho em fotos de lavoura não foi avaliado. Ferramenta de triagem e apoio à
decisão: não substitui o diagnóstico de um profissional habilitado.
