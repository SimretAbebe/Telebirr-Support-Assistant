import os
from google_play_scraper import reviews, Sort

result, _ = reviews(
    'cn.tydic.ethiopay',
    lang='en',
    country='us',
    sort=Sort.NEWEST,
    count=200
)

output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_reviews.txt'))
with open(output_path, 'w', encoding='utf-8') as f:
    for r in result:
        f.write(r['content'] + '\n---\n')

print(f"Saved {len(result)} reviews to {output_path}")