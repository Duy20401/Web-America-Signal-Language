import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize Firebase
sa_path = os.path.join('asl_web', 'firebase-service-account.json')
cred = credentials.Certificate(sa_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

print("=" * 60)
print("TESTING LETTERS DOCUMENT")
print("=" * 60)
letters_ref = db.collection('Vocabulary').document('UG8NXAPDdE23fMzgJSon')
letters_doc = letters_ref.get()

if letters_doc.exists:
    data = letters_doc.to_dict()
    print(f"Document exists! Found {len(data)} fields:")
    for key, val in data.items():
        val_preview = str(val)[:80] if val else "None"
        print(f"  '{key}': {val_preview}")
else:
    print("❌ Document DOES NOT EXIST")

print("\n" + "=" * 60)
print("TESTING DIGITS DOCUMENT")
print("=" * 60)
digits_ref = db.collection('Vocabulary').document('OUD3xJakGcN5JgMNqpnn')
digits_doc = digits_ref.get()

if digits_doc.exists:
    data = digits_doc.to_dict()
    print(f"Document exists! Found {len(data)} fields:")
    for key, val in data.items():
        val_preview = str(val)[:80] if val else "None"
        print(f"  '{key}': {val_preview}")
else:
    print("❌ Document DOES NOT EXIST")

print("\n" + "=" * 60)
print("ALL DOCUMENTS IN VOCABULARY COLLECTION")
print("=" * 60)
all_docs = db.collection('Vocabulary').stream()
doc_count = 0
for doc in all_docs:
    doc_count += 1
    print(f"\nDocument ID: {doc.id}")
    data = doc.to_dict()
    print(f"  Total fields: {len(data)}")
    print(f"  All field names: {sorted(data.keys())}")
    # Show a few sample values
    for i, (key, val) in enumerate(list(data.items())[:3]):
        val_preview = str(val)[:100] if val else "None"
        print(f"    {key}: {val_preview}")

if doc_count == 0:
    print("⚠️ No documents found in Vocabulary collection!")
else:
    print(f"\nTotal documents found: {doc_count}")
