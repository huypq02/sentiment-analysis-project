def rating_to_sentiment(rating):
    """
    Convert numeric rating (1-5) to sentiment label.
    
    :param rating: Numeric rating between 1-5
    :type rating: int
    :return: Sentiment label
    :rtype: str
    """
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:  # rating >= 4
        return 'Positive'
