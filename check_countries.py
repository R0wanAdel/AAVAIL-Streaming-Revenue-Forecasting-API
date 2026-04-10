from ingest import fetch_data

df = fetch_data()
countries = df['country'].unique()
print(f"Available countries ({len(countries)}):")
for c in sorted(countries):
    count = len(df[df['country'] == c])
    print(f"  {c}: {count} days of data")