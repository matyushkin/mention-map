import re
from datetime import date


# Common date patterns for diary/memoir texts
DATE_PATTERNS = [
    # "25 марта 1826 года", "25 марта 1826"
    r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})",
    # "March 25, 1826"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
    # "25.03.1826", "25/03/1826"
    r"(\d{1,2})[./](\d{1,2})[./](\d{4})",
    # "1826 год"
    r"(\d{4})\s+год",
]

RUSSIAN_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

ENGLISH_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def extract_dates(text: str) -> list[dict]:
    """Extract dates from text with their positions."""
    results = []

    # Russian dates
    for match in re.finditer(DATE_PATTERNS[0], text):
        day, month_name, year = match.groups()
        month = RUSSIAN_MONTHS.get(month_name)
        if month:
            results.append({
                "date": date(int(year), month, int(day)).isoformat(),
                "position": match.start(),
                "raw": match.group(),
            })

    # English dates
    for match in re.finditer(DATE_PATTERNS[1], text):
        month_name, day, year = match.groups()
        month = ENGLISH_MONTHS.get(month_name)
        if month:
            results.append({
                "date": date(int(year), month, int(day)).isoformat(),
                "position": match.start(),
                "raw": match.group(),
            })

    results.sort(key=lambda x: x["position"])
    return results
