"""
Generates the synthetic corpus at data/synthetic_documents.json.

Not part of the pipeline itself (the pipeline only ever reads the JSON file
this produces) -- kept as a separate, seeded script purely so the dataset is
reproducible and auditable rather than a hand-typed blob nobody can
regenerate. Contains only invented names/companies; no proprietary or real
company data.

Each of the ~15 underlying entities is referenced across several documents
using a mix of: full name (English), full name (Devanagari transliteration),
honorific + surname, initials/abbreviation, and a vague role-only
description with no name at all -- exactly the variation the entity
resolver is built to collapse back together.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_documents.json"

# Each entity: id, type, and a list of (surface_form, context_template) variants.
# {other} in a context template is filled with a random OTHER entity's name to
# create realistic co-occurrence signal across documents.
ENTITIES = [
    {
        "id": "GOLD_E01",
        "type": "person",
        "canonical": "Rajesh Kumar Sharma",
        "variants": [
            ("Rajesh Kumar Sharma", "{name} took over as regional sales director for the North zone last quarter, working closely with {other}."),
            ("राजेश कुमार शर्मा", "उत्तर क्षेत्र के लिए {name} को क्षेत्रीय बिक्री निदेशक नियुक्त किया गया, जो {other} के साथ मिलकर काम करते हैं।"),
            ("R.K. Sharma", "A meeting note filed by {name} discusses onboarding new sellers alongside {other}."),
            ("Mr. Sharma", "{name} approved the revised sales targets for the North zone this week."),
            ("the regional director for North zone", "According to the internal memo, {name} flagged a delay in the Delhi rollout involving {other}."),
        ],
    },
    {
        "id": "GOLD_E02",
        "type": "person",
        "canonical": "Priya Singh Chauhan",
        "variants": [
            ("Priya Singh Chauhan", "{name} leads the marketing team responsible for the new seller onboarding campaign with {other}."),
            ("प्रिया सिंह चौहान", "मार्केटिंग टीम की प्रमुख {name} ने {other} के साथ नई अभियान रणनीति पर चर्चा की।"),
            ("P.S. Chauhan", "Slide 4 of the campaign deck is credited to {name}, coordinating with {other}."),
            ("Ms. Chauhan", "{name} presented the Q3 marketing results to leadership."),
            ("the marketing head", "The campaign brief says {name} signed off on the creative direction with input from {other}."),
        ],
    },
    {
        "id": "GOLD_E03",
        "type": "person",
        "canonical": "Amit Patel",
        "variants": [
            ("Amit Patel", "{name} runs the Gujarat franchise operations and coordinates with {other}."),
            ("अमित पटेल", "गुजरात फ्रैंचाइज़ी के प्रमुख {name} ने {other} के साथ बैठक की।"),
            ("A. Patel", "Franchise report #12 is signed by {name}, referencing updates from {other}."),
            ("Mr. Patel", "{name} requested additional inventory support this month."),
            ("the Gujarat franchise owner", "A support ticket mentions that {name} escalated a billing issue also seen by {other}."),
        ],
    },
    {
        "id": "GOLD_E04",
        "type": "person",
        "canonical": "Sunita Devi Verma",
        "variants": [
            ("Sunita Devi Verma", "{name} is the finance officer overseeing reconciliation with {other}."),
            ("सुनीता देवी वर्मा", "वित्त अधिकारी {name} ने {other} के खातों की समीक्षा की।"),
            ("S.D. Verma", "The finance annexure was prepared by {name} in coordination with {other}."),
            ("Mrs. Verma", "{name} closed the monthly books two days early."),
            ("the finance officer", "An audit trail entry shows {name} approving a refund also flagged by {other}."),
        ],
    },
    {
        "id": "GOLD_E05",
        "type": "person",
        "canonical": "Vikram Aditya Rao",
        "variants": [
            ("Vikram Aditya Rao", "{name} was brought on as a technical consultant working alongside {other}."),
            ("विक्रम आदित्य राव", "तकनीकी सलाहकार {name} ने {other} के साथ सिस्टम की समीक्षा की।"),
            ("V.A. Rao", "The integration doc lists {name} as reviewer, alongside {other}."),
            ("Dr. Rao", "{name} recommended a change to the matching threshold."),
            ("the technical consultant", "A ticket notes {name} debugging an issue reported jointly with {other}."),
        ],
    },
    {
        "id": "GOLD_E06",
        "type": "person",
        "canonical": "Meena Kumari Yadav",
        "variants": [
            ("Meena Kumari Yadav", "{name} manages HR onboarding and works with {other} on interviews."),
            ("मीना कुमारी यादव", "एचआर प्रबंधक {name} ने {other} के साथ साक्षात्कार लिए।"),
            ("M.K. Yadav", "The hiring summary was drafted by {name}, cc'ing {other}."),
            ("Ms. Yadav", "{name} finalized three offer letters this week."),
            ("the HR manager", "According to the note, {name} coordinated a joint session with {other}."),
        ],
    },
    {
        "id": "GOLD_E07",
        "type": "person",
        "canonical": "Suresh Chandra Gupta",
        "variants": [
            ("Suresh Chandra Gupta", "{name} is the senior accountant handling ledgers shared with {other}."),
            ("सुरेश चंद्र गुप्ता", "वरिष्ठ लेखाकार {name} ने {other} के साथ खाता बही की जांच की।"),
            ("S.C. Gupta", "Ledger review #7 was signed off by {name}, referencing {other}."),
            ("Shri Gupta", "{name} identified a discrepancy in the March invoices."),
            ("the senior accountant", "A note says {name} cross-checked figures also reviewed by {other}."),
        ],
    },
    {
        "id": "GOLD_E08",
        "type": "org",
        "canonical": "KCD Traders Pvt Ltd",
        "variants": [
            ("KCD Traders Pvt Ltd", "{name} placed a bulk order this month, coordinated through {other}."),
            ("केसीडी ट्रेडर्स प्राइवेट लिमिटेड", "{name} ने इस महीने थोक ऑर्डर दिया, जिसे {other} के माध्यम से संभाला गया।"),
            ("KCD Traders", "Invoice #4471 was raised for {name}, processed with {other}."),
            ("KCD", "{name} renewed its paid seller subscription."),
            ("the NCR-based paid seller", "A KCD dataset entry lists {name} as active, cross-referenced with {other}."),
        ],
    },
    {
        "id": "GOLD_E09",
        "type": "org",
        "canonical": "Bharat Steel Industries",
        "variants": [
            ("Bharat Steel Industries", "{name} supplies raw steel to buyers connected via {other}."),
            ("भारत स्टील इंडस्ट्रीज", "{name} ने {other} के माध्यम से नया ऑर्डर प्राप्त किया।"),
            ("BSI", "Shipment log references {name}, coordinated with {other}."),
            ("Bharat Steel", "{name} confirmed the delivery timeline for next week."),
            ("the steel supplier from Ludhiana", "A procurement note lists {name} alongside {other}."),
        ],
    },
    {
        "id": "GOLD_E10",
        "type": "org",
        "canonical": "Nagpur Agro Exports Ltd",
        "variants": [
            ("Nagpur Agro Exports Ltd", "{name} is expanding its export pipeline with {other}."),
            ("नागपुर एग्रो एक्सपोर्ट्स लिमिटेड", "{name} ने {other} के साथ निर्यात समझौता किया।"),
            ("NAE", "The export manifest names {name}, alongside {other}."),
            ("Nagpur Agro", "{name} shipped its first container of the season."),
            ("the agro exports company from Nagpur", "A trade note mentions {name} working with {other}."),
        ],
    },
    {
        "id": "GOLD_E11",
        "type": "org",
        "canonical": "Global Textile Solutions",
        "variants": [
            ("Global Textile Solutions", "{name} onboarded as a paid seller, referred by {other}."),
            ("ग्लोबल टेक्सटाइल सॉल्यूशंस", "{name} को {other} द्वारा संदर्भित किया गया।"),
            ("GTS", "Account record shows {name} upgraded its plan, linked to {other}."),
            ("Global Textile", "{name} requested a catalog review this week."),
            ("the textile solutions provider", "A ticket references {name} alongside {other}."),
        ],
    },
    {
        "id": "GOLD_E12",
        "type": "org",
        "canonical": "Chandrashekhar Industries",
        "variants": [
            ("Chandrashekhar Industries", "{name} filed a compliance update coordinated with {other}."),
            ("चंद्रशेखर इंडस्ट्रीज", "{name} ने {other} के साथ अनुपालन रिपोर्ट प्रस्तुत की।"),
            ("Chandrasekar Industries", "A regional filing lists {name}, cross-checked against {other}."),
            ("CI", "{name} updated its GST details this quarter."),
            ("the industrial supplier with the spelling-variant name", "Internal notes mention {name} alongside {other}."),
        ],
    },
    {
        "id": "GOLD_E13",
        "type": "org",
        "canonical": "Himalayan Traders & Co.",
        "variants": [
            ("Himalayan Traders & Co.", "{name} negotiated a new rate card with {other}."),
            ("हिमालयन ट्रेडर्स एंड कंपनी", "{name} ने {other} के साथ नई दर तय की।"),
            ("HT&Co", "The rate card memo lists {name}, alongside {other}."),
            ("Himalayan Traders", "{name} confirmed acceptance of the revised terms."),
            ("the hill-region trading company", "A note references {name} working with {other}."),
        ],
    },
    {
        "id": "GOLD_E14",
        "type": "org",
        "canonical": "Rani Enterprises",
        "variants": [
            ("Rani Enterprises", "{name} is a Jaipur-based enterprise partnered with {other}."),
            ("रानी एंटरप्राइजेज", "{name} ने {other} के साथ साझेदारी की।"),
            ("Rani Ent.", "The partnership brief lists {name}, alongside {other}."),
            ("Rani", "{name} signed the renewal agreement yesterday."),
            ("the Jaipur-based enterprise", "A sourcing note mentions {name} together with {other}."),
        ],
    },
    {
        "id": "GOLD_E15",
        "type": "org",
        "canonical": "Sagar Logistics Pvt Ltd",
        "variants": [
            ("Sagar Logistics Pvt Ltd", "{name} handles last-mile delivery for shipments involving {other}."),
            ("सागर लॉजिस्टिक्स प्राइवेट लिमिटेड", "{name} ने {other} से जुड़ी खेप संभाली।"),
            ("SLPL", "The delivery log lists {name}, coordinated with {other}."),
            ("Sagar Logistics", "{name} confirmed same-day delivery for the Mumbai route."),
            ("the logistics partner from Mumbai", "A dispatch note mentions {name} alongside {other}."),
        ],
    },
]


def build_documents():
    documents = []
    doc_counter = 0
    n = len(ENTITIES)
    # Each entity is given a FIXED "associate" entity (a colleague/partner
    # organization that keeps recurring across that entity's documents),
    # rather than a random one picked per-document. A fixed offset (mod n)
    # gives every entity a distinct, consistent associate. This is what
    # makes co-occurrence a genuine signal: entity X's mentions consistently
    # co-occur with the SAME other entity across documents, while unrelated
    # entities essentially never share an associate. A per-document random
    # associate (an earlier version of this script) produced spurious
    # one-off overlaps that caused unrelated entities to look connected --
    # see README "Design Decisions" for why this matters.
    associate_by_id = {
        ENTITIES[i]["id"]: ENTITIES[(i + 6) % n] for i in range(n)
    }
    for entity in ENTITIES:
        associate = associate_by_id[entity["id"]]
        associate_surface = associate["canonical"]
        # Use 4 of the 5 authored variants per entity (full English name,
        # Devanagari form, abbreviation, vague role-only description) to land
        # the corpus size in the 40-60 document range requested by the spec.
        selected_variants = [entity["variants"][i] for i in (0, 1, 2, 4)]
        for surface_form, template in selected_variants:
            other_surface = associate_surface
            context = template.format(name=surface_form, other=other_surface)
            doc_counter += 1
            doc_id = f"doc_{doc_counter:03d}"
            documents.append(
                {
                    "doc_id": doc_id,
                    "text": context,
                    "mentions": [
                        {
                            "surface_form": surface_form,
                            "context": context,
                            "gold_entity_id": entity["id"],
                        },
                        {
                            "surface_form": other_surface,
                            "context": context,
                            "gold_entity_id": associate["id"],
                        },
                    ],
                }
            )
    random.shuffle(documents)
    # re-number doc_ids after shuffle so IDs don't leak generation order
    for i, doc in enumerate(documents, start=1):
        doc["doc_id"] = f"doc_{i:03d}"
    return documents


def main():
    documents = build_documents()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
    total_mentions = sum(len(d["mentions"]) for d in documents)
    print(f"Wrote {len(documents)} documents ({total_mentions} mentions) to {OUT_PATH}")


if __name__ == "__main__":
    main()
