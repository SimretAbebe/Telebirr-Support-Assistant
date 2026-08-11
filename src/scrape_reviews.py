from google_play_scraper import reviews, Sort

result, _ = reviews(
    'cn.tydic.ethiopay',
    lang='en',
    country='us',
    sort=Sort.NEWEST,
    count=200
)

with open('data/raw_reviews.txt', 'w', encoding='utf-8') as f:
    for r in result:
        f.write(r['content'] + '\n---\n')

print(f"Saved {len(result)} reviews to data/raw_reviews.txt")