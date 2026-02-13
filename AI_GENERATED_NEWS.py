import csv
import random
from faker import Faker

fake = Faker()

# Configuration
NUM_ARTICLES = 20000
FILE_NAME = "ai_generated_news_20k.csv"

def generate_tech_news():
    company = fake.company()
    tech = random.choice(["AI-driven", "Quantum", "Blockchain", "SaaS", "Cloud-native", "Edge-computing"])
    product = random.choice(["processor", "platform", "algorithm", "interface", "encryption", "neural-net"])
    
    s1 = f"{company} has officially launched its new {tech} {product} today."
    s2 = f"The technology aims to streamline {fake.bs()} across global enterprise networks."
    s3 = f"Early benchmarks suggest a {random.randint(20, 80)}% increase in operational efficiency compared to legacy systems."
    s4 = f"Industry analysts predict this will become the new standard for {fake.word()} integration by 2027."
    s5 = f"Lead developer {fake.name()} stated that the project has been in stealth mode for over three years."
    s6 = f"Investors have already poured an additional ${random.randint(10, 500)} million into the scale-up phase."
    s7 = f"However, cybersecurity experts warn that the {product} may require new protocols to prevent {fake.word()} leaks."
    s8 = f"Beta testing in {fake.city()} revealed that the {tech} system adapts to user behavior in real-time."
    s9 = f"The company plans to release an open-source version of the API to encourage third-party development."
    s10 = f"Market shares for {company} rose sharply by {random.uniform(1.5, 9.0):.1f}% following the press conference."
    return " ".join([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])

def generate_health_news():
    research_org = f"The {fake.city()} Medical Institute"
    condition = random.choice(["chronic fatigue", "heart disease", "insomnia", "migraines", "Type 2 diabetes"])
    
    s1 = f"New clinical trials at {research_org} show promising results for treating {condition}."
    s2 = f"The study involved over {random.randint(1000, 5000)} volunteers over a six-month double-blind period."
    s3 = f"Researchers found that a combination of {fake.word()} and lifestyle changes led to significant recovery rates."
    s4 = f"Health officials are expected to review the findings for updated national guidelines next month."
    s5 = f"Dr. {fake.name()}, the lead researcher, noted that the results exceeded all preliminary expectations."
    s6 = f"The data indicates that {random.randint(70, 95)}% of patients experienced a reduction in symptoms within weeks."
    s7 = f"Despite the success, some independent doctors are calling for a longer-term follow-up study."
    s8 = f"Funding for this massive medical project was provided by a grant from the {fake.city()} Foundation."
    s9 = f"Patient advocacy groups have already started lobbying for faster insurance coverage of the treatment."
    s10 = f"If approved, this could become the primary line of defense against {condition} worldwide."
    return " ".join([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])

def generate_finance_news():
    currency = random.choice(["USD", "Bitcoin", "The Euro", "Global markets", "The Yen"])
    
    s1 = f"{currency} saw a significant shift following the latest report from the central bank."
    s2 = f"Investors are reacting to news regarding {fake.catch_phrase().lower()} affecting international trade."
    s3 = f"The market closed with a {random.uniform(0.5, 4.0):.2f}% change in major indices across the board."
    s4 = f"Financial experts suggest maintaining a diverse portfolio to hedge against upcoming market volatility."
    s5 = f"The CEO of {fake.company()} warned that inflationary pressures are not as 'transitory' as once thought."
    s6 = f"Commodity prices, particularly in the {fake.word()} sector, reached a five-year high today."
    s7 = f"Rumors of a potential merger between {fake.company()} and {fake.company()} added to the trading frenzy."
    s8 = f"The labor department is scheduled to release the next employment figures this coming Friday."
    s9 = f"Retail investors have increased their positions in {random.choice(['Gold', 'Tech Stocks', 'Real Estate'])} significantly."
    s10 = f"Economists remain divided on whether the current trend suggests a soft landing or a deeper recession."
    return " ".join([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])

def generate_env_news():
    location = fake.country()
    issue = random.choice(["carbon emissions", "plastic waste", "solar energy", "reforestation", "ocean acidity"])
    
    s1 = f"Government leaders in {location} have pledged to drastically reduce {issue} by 2030."
    s2 = f"The initiative involves a multi-billion dollar investment into {fake.bs()} infrastructure."
    s3 = f"Environmental activists have largely praised the move as a historic turning point for the region."
    s4 = f"However, some critics argue that the timeline for these changes is not nearly aggressive enough."
    s5 = f"The plan includes the installation of {random.randint(500, 2000)} new monitoring stations across {location}."
    s6 = f"Local wildlife populations, particularly {fake.word()}s, are expected to benefit from the habitat restoration."
    s7 = f"Satellite data suggests that current {issue} levels have already plateaued for the first time in a decade."
    s8 = f"Innovative tech from {fake.company()} is being deployed to capture and store excess waste."
    s9 = f"Community leaders are organizing workshops to help citizens transition to more sustainable practices."
    s10 = f"Global climate observers are looking at {location} as a potential blueprint for other developing nations."
    return " ".join([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])

# Execution logic
domains = [generate_tech_news, generate_health_news, generate_finance_news, generate_env_news]
domain_names = ["Technology", "Healthcare", "Finance", "Environment"]

print(f"Creating {NUM_ARTICLES} articles. This may take a few seconds...")

with open(FILE_NAME, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Category', 'Article_Body'])
    for i in range(1, NUM_ARTICLES + 1):
        idx = random.randint(0, 3)
        writer.writerow([i, domain_names[idx], domains[idx]()])

print(f"Successfully saved {NUM_ARTICLES} articles to {FILE_NAME}!")