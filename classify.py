def classify_text(text):
    text = text.lower()
    if "invoice" in text:
        return "Invoice"
    elif "receipt" in text:
        return "Receipt"
    elif "school" in text:
        return "School Document"
    else:
        return "Other"
