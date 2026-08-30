import os
import pandas as pd

file_path=os.path.join('..','data','temporal_graph',"CollegeMsg","raw_CollegeMsg.txt")

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    names=["u", "i", "ts"]
)

# Unix timestamp → day
df["day"] = df["ts"] // 86400

# 같은 날 동일 방향 (u, i) 중복 제거
daily_df = df.drop_duplicates(
    subset=["u", "i", "day"]
)

print("Nodes:", len(set(daily_df["u"]) | set(daily_df["i"])))
print("Edge events:", len(daily_df))
print("Days:", daily_df["day"].nunique())