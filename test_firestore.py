"""Test script to check Firestore data"""
import os
import sys
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
sa_path = os.path.join('asl_web', 'firebase-service-account.json')
cred = credentials.Certificate(sa_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

print("=" * 60)
print("LISTING ALL DOCUMENTS IN 'Vocabulary' COLLECTION")
print("=" * 60)
coll = db.collection('Vocabulary')
docs = coll.stream()
doc_list = list(docs)
print(f"Found {len(doc_list)} documents:")
for doc in doc_list:
    print(f"  Document ID: '{doc.id}'")
    data = doc.to_dict()
    print(f"    Fields: {list(data.keys())[:5]}...")  # Show first 5 fields
print()

print("=" * 60)
print("TESTING LETTERS DOCUMENT")
print("=" * 60)
doc_ref = db.collection('Vocabulary').document('UG8NXAPDdE23fMzgJSon')
doc = doc_ref.get()

if doc.exists:
    data = doc.to_dict()
    print(f"Document exists! Found {len(data)} fields:")
    for key, val in data.items():
        if isinstance(val, str):
            print(f"  '{key}' -> {val[:80]}...")
        else:
            print(f"  '{key}' -> {type(val).__name__}")
else:
    print("Document DOES NOT EXIST!")

print("\n" + "=" * 60)
print("TESTING DIGITS DOCUMENT")
print("=" * 60)
doc_ref = db.collection('Vocabulary').document('OUD3xJakGcN5JgMNqpnn')
doc = doc_ref.get()

if doc.exists:
    data = doc.to_dict()
    print(f"Document exists! Found {len(data)} fields:")
    for key, val in data.items():
        if isinstance(val, str):
            print(f"  '{key}' -> {val[:80]}...")
        else:
            print(f"  '{key}' -> {type(val).__name__}")
else:
    print("Document DOES NOT EXIST!")
