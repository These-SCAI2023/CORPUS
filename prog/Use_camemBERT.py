from optparse import OptionParser
import re
import glob
from pathlib import Path
import json
import os
import csv
from transformers import CamembertTokenizer, CamembertForTokenClassification, pipeline, Pipeline, TokenClassificationPipeline
import transformers
# import torch
from itertools import chain


def get_parser():
    """Returns a command line parser
    Returns:
        OptionParser. The command line parser
    """
    parser = OptionParser()
    parser.add_option("-d", "--data_path", dest="data_path",
                      help="""Chemin vers les fichiers txt (exemple DATA/*)""", type="string", default="../DATA/")
    parser.add_option('-F', '--Force', help='Recalculer les distances même si déjà faites',
                      action='store_true', default=False)
    return parser


parser = get_parser()
options, _ = parser.parse_args()
path_corpora = options.data_path
print("")
print("-"*40)
print(f"Path corpora : '{path_corpora}'")
print("--> pour spécifier un autre chemin utiliser l'option -d")
print("-"*40)


def lire_fichier(chemin, is_json=False):
    with open(chemin, encoding='utf-8') as f:
        if not is_json:
            return f.read()
        else:
            return json.load(f)


def stocker(chemin, contenu, is_json=False, verbose=False):
    with open(chemin, "w", encoding='utf-8') as w:
        if is_json:
            w.write(json.dumps(contenu, indent=2, ensure_ascii=False))
        else:
            w.write(contenu)
    if verbose:
        print(f"  Output written in {chemin}")


def chunk_text(text, chunk_size, overlap):
    """Splits text into chunks of specified size with overlap."""
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


def get_entity_dict(entity: dict) -> dict:
    return {"label": entity['entity_group'],
            "text": entity['word'],
            "jalons": [entity['start'], entity['end']]}


def dico_resultats(texte, nlp: TokenClassificationPipeline, chunk_size=512, overlap=50) -> dict[str, dict]:
    """returns a dict of all the entitites in the text like: {entity_0: {"label": PER, "text": "Harry James Potter", "jalons": [0, 30]}}
    """
    chunks: list = chunk_text(texte, chunk_size, overlap)
    all_ner_results: list[dict] = chain.from_iterable(
        [nlp(chunk) for chunk in chunks])
    return {f"entite_{i}": get_entity_dict(
        entity=ent) for i, ent in enumerate(all_ner_results)}


def generate_bio_tags(entity: dict, sep: str = " ") -> list[list[str]]:
    """generate bio tags (B/I) for potentially multiword entities.
    """
    words: list[str] = entity["word"].split(sep)
    bio_entity_list: list[list] = [
        [words.pop(0), f"B-{entity['entity_group']}"]]
    bio_entity_list.extend(
        [[word, f"I-{entity['entity_group']}"] for word in words])
    return bio_entity_list


def bio_camemBERT(texte, nlp: TokenClassificationPipeline, chunk_size=512, overlap=50) -> list[list[str]]:
    """returns a list of bio entities for the text like: [["Harry", "B", "PER"], ["James", "I", "PER"], ["Potter", "I", "PER"]]
    """
    chunks: list = chunk_text(texte, chunk_size, overlap)
    all_ner_results: list[dict] = chain.from_iterable(
        [nlp(chunk) for chunk in chunks])
    return chain.from_iterable(
        [generate_bio_tags(entity=entity) for entity in all_ner_results])


if __name__ == "__main__":
    model_name = "Jean-Baptiste/camembert-ner-with-dates"
    nlp: TokenClassificationPipeline = pipeline("ner", model=model_name, tokenizer=model_name,
                                                aggregation_strategy="simple", device=-1)

    liste_subcorpus = list(Path(path_corpora).glob("*"))
    print(liste_subcorpus)
    print(os.getcwd())
    if len(liste_subcorpus) == 0:
        print(f"Pas de dossier trouvé dans {path_corpora}, traitement terminé")
        exit()

    for subcorpus in liste_subcorpus:
        print("  Processing %s" % subcorpus)
        liste_txt = glob.glob("%s/*_REF/*.txt" % subcorpus)
        liste_txt += glob.glob("%s/OCR/*/*.txt" % subcorpus)
        print("  nombre de fichiers txt trouvés :", len(liste_txt))
        for path in liste_txt:
            dossiers = re.split("/", path)[:-1]
            nom_txt = re.split("/", path)[-1]
            os.makedirs(os.path.join(*dossiers, "NER"), exist_ok=True)
            path_output = os.path.join(
                *dossiers, "NER", f"{nom_txt}_camembert-{transformers.__version__}.json")
            path_output_bio = os.path.join(
                *dossiers, "NER", f"{nom_txt}_camembert-{transformers.__version__}.bio")
            if os.path.exists(path_output) and not options.Force:
                print("Already DONE : ", path_output)
                continue

            texte = lire_fichier(path)
            entites = dico_resultats(texte, nlp)
            stocker(path_output, entites, is_json=True)

            entites_bio = bio_camemBERT(texte, nlp)
            with open(path_output_bio, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=' ', quotechar='|')
                writer.writerows(entites_bio)
