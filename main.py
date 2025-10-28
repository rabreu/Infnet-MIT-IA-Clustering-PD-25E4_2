import pandas as pd

dataset = pd.read_csv("datasets/Country-data.csv")

country_data_df = pd.DataFrame(dataset)

print(country_data_df.head())
print(country_data_df.info())

# country_data_df['child_mort'] = country_data_df['child_mort'].apply(lambda x: "%.2f" % float(x))
# country_data_df['exports'] = country_data_df['exports'].apply(lambda x: "%.2f" % float(x))
# country_data_df['health'] = country_data_df['health'].apply(lambda x: "%.2f" % float(x))
# country_data_df['imports'] = country_data_df['imports'].apply(lambda x: "%.2f" % float(x))
# country_data_df['inflation'] = country_data_df['inflation'].apply(lambda x: "%.2f" % float(x))
# country_data_df['life_expec'] = country_data_df['life_expec'].apply(lambda x: "%.2f" % float(x))
# country_data_df['total_fer'] = country_data_df['total_fer'].apply(lambda x: "%.2f" % float(x))

country_data_df['income'] = country_data_df['income'].apply(lambda x: float(x))
country_data_df['gdpp'] = country_data_df['gdpp'].apply(lambda x: float(x))

print(country_data_df.head())
print(country_data_df.info())