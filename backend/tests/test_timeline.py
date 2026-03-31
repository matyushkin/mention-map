from timeline.extractor import extract_dates


def test_russian_date():
    text = "Запись от 25 марта 1826 года. Сегодня видел Пушкина."
    dates = extract_dates(text)
    assert len(dates) == 1
    assert dates[0]["date"] == "1826-03-25"


def test_english_date():
    text = "Entry from March 25, 1826. Met with Byron today."
    dates = extract_dates(text)
    assert len(dates) == 1
    assert dates[0]["date"] == "1826-03-25"


def test_multiple_dates():
    text = (
        "1 января 1820 года начался новый год. "
        "15 июня 1820 года мы отправились в путь."
    )
    dates = extract_dates(text)
    assert len(dates) == 2
    assert dates[0]["date"] == "1820-01-01"
    assert dates[1]["date"] == "1820-06-15"


def test_no_dates():
    text = "Просто текст без дат."
    dates = extract_dates(text)
    assert len(dates) == 0
