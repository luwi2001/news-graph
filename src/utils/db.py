import sqlite3
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone, timedelta
import email.utils

DB_PATH = "~/services/rsstt/config/db.sqlite3"

# 系统本地时区偏移 (CST +08:00)
LOCAL_TZ = timezone(timedelta(hours=8))


def parse_datetime(s: Optional[str]) -> Optional[datetime]:
    """解析 SQLite 日期字符串为 UTC naive datetime 对象"""
    if s is None:
        return None
    try:
        # 尝试 ISO 格式
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            dt = dt.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        try:
            # 尝试 SQLite 默认格式 YYYY-MM-DD HH:MM:SS (视为本地时间)
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            try:
                # 尝试 RFC 2822 格式 (Wed, 29 Apr 2026 23:58:12 -0700)
                dt = email.utils.parsedate_to_datetime(s)
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt
            except Exception:
                return None


@dataclass
class NewsEntry:
    id: int
    feed_id: int
    title: str
    link: str
    pub_date: Optional[datetime]
    description: Optional[str]
    fetched_at: datetime


def load_entries(conn: sqlite3.Connection, limit: Optional[int] = None) -> list[NewsEntry]:
    """从数据库加载新闻条目"""
    query = """
        SELECT id, feed_id, title, link, pub_date, description, fetched_at
        FROM entries
        WHERE pub_date IS NOT NULL
        ORDER BY pub_date DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)
    rows = cursor.fetchall()

    entries = []
    for row in rows:
        pub_date = parse_datetime(row[4])
        fetched_at = parse_datetime(row[6]) or datetime.now()
        entries.append(NewsEntry(
            id=row[0],
            feed_id=row[1],
            title=row[2],
            link=row[3],
            pub_date=pub_date,
            description=row[5],
            fetched_at=fetched_at
        ))
    return entries


def get_connection(db_path: str) -> sqlite3.Connection:
    """获取数据库连接"""
    import os
    db_path = os.path.expanduser(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn