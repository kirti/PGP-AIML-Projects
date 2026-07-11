"""
Synthetic 'medical manual' PDF generator.
Produces a placeholder reference document with the same general structure as
a medical manual (disease entries: overview / symptoms / treatment), using
entirely made-up, non-copyrighted content -- NOT excerpted from any real
medical reference (e.g. not the Merck Manual). Safe to use for testing a
RAG (retrieval-augmented generation) pipeline's chunking/embedding/retrieval
mechanics without exposing real copyrighted or medical content.
"""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("/mnt/user-data/outputs/dummy_medical_manual.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Sample Reference Manual (Synthetic / Placeholder Content)", styles['Title']))
story.append(Spacer(1, 12))
story.append(Paragraph(
    "This document contains entirely fictional, AI-generated placeholder content. "
    "It does not reproduce or paraphrase any real medical reference (e.g. the Merck "
    "Manual). It exists solely to let a retrieval-augmented-generation (RAG) pipeline "
    "be smoke-tested end-to-end (PDF loading, chunking, embedding, vector search, and "
    "generation) without using real copyrighted or clinical content.",
    styles['Normal']))
story.append(PageBreak())

# A set of entirely made-up "conditions" with fictional names, so nothing here
# could be mistaken for real medical guidance.
fake_conditions = [
    ("Zentaris Syndrome", "A fictional placeholder condition affecting the sample "
     "connective tissue in test-patients.",
     "Symptoms include elevated Placeholder-Marker-7 levels, mild fatigue-analog "
     "readings, and irregular test-signal patterns.",
     "Treatment in this fictional scenario involves Compound A-113 administered "
     "twice daily, alongside supportive placeholder therapy."),
    ("Vindral Fever", "A fictional placeholder febrile condition used for testing "
     "document retrieval.",
     "Symptoms include simulated temperature elevation, sample joint discomfort, "
     "and placeholder rash patterns.",
     "Treatment involves fictional agent Rendozol 250mg every 8 hours for 5 "
     "placeholder days."),
    ("Korvath's Placeholder Disorder", "A fictional metabolic placeholder condition.",
     "Symptoms include simulated appetite changes, placeholder energy fluctuation, "
     "and sample lab-marker deviation.",
     "Management includes fictional dietary placeholder adjustment and Compound "
     "B-207 as needed."),
    ("Thelmond Reflex Anomaly", "A fictional neurological placeholder finding used "
     "purely for RAG-pipeline testing.",
     "Symptoms include simulated reflex-timing variance and placeholder "
     "coordination test results.",
     "Fictional first-line management is placeholder physical therapy; fictional "
     "second-line is Compound C-9 as an adjunct."),
    ("Braxen Pulmonary Placeholder Index", "A fictional respiratory placeholder "
     "marker referenced across several sample entries.",
     "Symptoms include simulated breath-rate deviation and placeholder oxygen-index "
     "readings.",
     "Fictional treatment includes placeholder inhaled Compound D-4 and supportive "
     "monitoring."),
]

for name, overview, symptoms, treatment in fake_conditions:
    story.append(Paragraph(name, styles['Heading1']))
    story.append(Paragraph("Overview: " + overview, styles['Normal']))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Symptoms: " + symptoms, styles['Normal']))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Treatment: " + treatment, styles['Normal']))
    story.append(Spacer(1, 18))

doc.build(story)
print("PDF created")
