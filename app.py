
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

## Load data set
df=pd.read_csv("/Users/amalthomas/Desktop/Email_classifier/combined_emails_with_natural_pii.csv")

def split_subject_and_body(text):
    lines = str(text).split('\n', 1)
    subject = lines[0].strip()
    body = lines[1].strip() if len(lines) > 1 else ''
    return pd.Series([subject, body])

# Apply to your column (replace 'subject_body' with the actual column name)
df[['subject', 'body']] = df['email'].apply(split_subject_and_body)

#editted 1 - final
import re

def mask_pii_pci(text: str):
    patterns = {
        "full_name": r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone_number": r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b",
        "dob": r"\b(?:\d{1,2}[\/\-]){2}\d{2,4}\b",
        "aadhar_num": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "credit_debit_no": r"\b(?:\d[ -]*?){13,16}\b",
        "cvv_no": r"\b\d{3}\b",
        "expiry_no": r"\b(0[1-9]|1[0-2])[\/\-]?\d{2,4}\b",
    }

    entities = []
    masked_text = text
    offset = 0  # To keep track of character shifts due to masking

    for entity_name, pattern in patterns.items():
        for match in re.finditer(pattern, masked_text):
            start = match.start()
            end = match.end()
            entity_text = match.group(0)

            # Adjust start and end with current offset
            adjusted_start = start + offset
            adjusted_end = end + offset
            
            # Create masked version
            replacement = f"[{entity_name}]"
            shift = len(replacement) - len(entity_text)
            offset += shift

            # Replace entity in masked text (once, at the current match only)
            masked_text = masked_text[:start] + replacement + masked_text[end:]

            entities.append({
                "position": [start, start + len(replacement)],
                "classification": entity_name,
                "entity": entity_text
            })

    return masked_text, entities
df['masked_email_text'] = df['body'].apply(lambda x: mask_pii_pci(x)[0])

# Define rule-based labeling function
def categorize_email(text):
    #text = text.lower()
    if any(word in text for word in ['billing', 'invoice', 'charge', 'refund', 'payment','overcharged','transaction']):
        return 'Billing Issues'
    elif any(word in text for word in ['account', 'login', 'reset', 'password', 'username','change email','profile','personal info']):
        return 'Account Management'
    elif any(word in text for word in ['crash', 'bug', 'issue', 'error', 'not working','failed','glitch','connection','API down']):  
        return 'Technical Support'
    elif any(word in text for word in ['product info','features','specifications','how does it work','demo','trial']):
        return 'Product Enquiries'
    elif any(word in text for word in ['tracking','delayed','order not received','delivery','shipping status','wrong item']):
        return 'Shipping & Delivery'
    elif any(word in text for word in ['cancel subscription','stop service','end membership','unsubscribe','discontinue']):
        return 'Cancellation Request'
    elif any(word in text for word in ['question','need help','assistance','contact','customer service']):
        return 'General Inquiry'
    elif any(word in text for word in ['suggestion','feedback','review','opinion','experience','comment']):
        return 'Feedback'
    elif any(word in text for word in ['privacy','GDPR','terms of service','copyright','policy','legal']):
        return 'Legal & Compliance'
    else:
        return 'Other'

# Apply rule to generate labels
df['category'] = df['masked_email_text'].apply(categorize_email)

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
df['masked_email_text'] = df['body'].apply(lambda x: mask_pii_pci(x)[0])



X_train, X_test, y_train, y_test = train_test_split(df['masked_email_text'], df['category'], test_size=0.2, random_state=42)

pipeline = make_pipeline(
    
    TfidfVectorizer(max_features=20000, stop_words='english'),
    LinearSVC()
)

pipeline.fit(X_train, y_train)

import gradio as gr
import json

def parse_input(input_str):
    try:
        input_data = json.loads(input_str)
        return input_data.get("input_email_body", "")
    except (json.JSONDecodeError, AttributeError):
        return input_str  # Treat it as plain text

def generate_json_output(Email):
    try:
        text = parse_input(Email)

        if not text.strip():
            return json.dumps({"error": "No email content provided."}, indent=2)

        # Masking step
        masked_text, entities = mask_pii_pci(text)

        # Prediction step
        category = pipeline.predict([masked_text])[0]

        output = {
            "input_email_body": text,
            "list_of_masked_entities": entities,
            "masked_email": masked_text,
            "category_of_the_email": category
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

          
#gr.Interface(fn=generate_json_output, inputs="text", outputs="text").launch()
gr.Interface(
    fn=generate_json_output,
    inputs=gr.Textbox(
        lines=12,
        label="Input Email (Text or JSON)",
        placeholder=(
            "You can input:\n"
            "1️⃣ Plain Text: Hi John, contact me at john.doe@example.com or call 123-456-7890.\n"
            "2️⃣ JSON: {\n  \"input_email_body\": \"Hi John, contact me at john.doe@example.com or call 123-456-7890.\"\n}"
        )
    ),
    outputs=gr.Textbox(label="Output (JSON)"),
    title="📧 Email Classification and PII Masking",
    description="Paste either a plain email body or a JSON with an 'input_email_body'. The system will return masked and classified results."
).launch()