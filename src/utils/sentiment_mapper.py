def rating_to_sentiment(rating):
    """Convert numeric rating (1-5) to sentiment label.
    
    Args:
        rating: Numeric rating between 1-5
        
    Returns:
        str: 'Negative', 'Neutral', or 'Positive'
    """
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:  # rating >= 4
        return 'Positive'
