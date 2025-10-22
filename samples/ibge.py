import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

uri_censo_basico_br = 'https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/Agregados_por_Municipio_csv/Agregados_por_municipios_basico_BR_20250417.zip'

ibge_censo_basico_br = pd.read_csv(uri_censo_basico_br, encoding='ISO-8859-1', sep=';')

ibge_censo_basico_br = ibge_censo_basico_br[['CD_MUN', 'NM_MUN', 'NM_UF', 'NM_REGIAO', 'v0001', 'v0002', 'v0003', 'v0004', 'v0005', 'v0006', 'v0007']]

censo_basico_sp = ibge_censo_basico_br.query("NM_UF == 'São Paulo'").reset_index(drop=True)
# print(censo_basico_sp.head())
#
# print(censo_basico_sp.info())

censo_basico_sp['v0005'] = censo_basico_sp['v0005'].apply(lambda v5: float(v5.replace(',', '.')))
censo_basico_sp['v0006'] = censo_basico_sp['v0006'].apply(lambda v6: float(v6.replace(',', '.')))

# print(censo_basico_sp.info())

sp_kmeans = KMeans(n_clusters=3, random_state=42).fit(censo_basico_sp.iloc[:, 4:])
censo_basico_sp['cluster'] = sp_kmeans.labels_
print(censo_basico_sp['cluster'].value_counts())

sp_url = 'https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2023/UFs/SP/SP_Municipios_2023.zip'
sp_geodf = gpd.read_file(sp_url)
sp_geodf = sp_geodf[['CD_MUN', 'geometry']]
sp_geodf['CD_MUN'] = sp_geodf['CD_MUN'].astype(int)
sp_geodf.plot()

plt.show()

df_join = pd.merge(sp_geodf, censo_basico_sp, on='CD_MUN', how='inner')

print(df_join.head())
df_join.plot(column='cluster', legend=True)

plt.show()