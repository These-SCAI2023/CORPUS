from optparse import OptionParser
import re
import glob
from pathlib import Path
import json
import os
import csv
from transformers import CamembertTokenizer, CamembertForTokenClassification, pipeline
import torch


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


def dico_resultats(texte, nlp, chunk_size=512, overlap=50):
    all_ner_results = []
    for chunk in chunk_text(texte, chunk_size, overlap):
        ner_results = nlp(chunk)
        all_ner_results.extend(ner_results)

    dico_resultats = {}
    for i, ent in enumerate(all_ner_results):
        entite = f"entite_{i}"
        dico_resultats[entite] = {
            "label": ent['entity_group'],
            "text": ent['word'],
            "jalons": [ent['start'], ent['end']]
        }
    return dico_resultats


def bio_spacy(texte, nlp, chunk_size=512, overlap=50):
    all_ner_results = []
    for chunk in chunk_text(texte, chunk_size, overlap):
        ner_results = nlp(chunk)
        all_ner_results.extend(ner_results)

    liste_bio = []
    for ent in all_ner_results:
        liste_bio.append([ent['word'], ent['entity_group']])
    return liste_bio


if __name__ == "__main__":
    model_name = "Jean-Baptiste/camembert-ner-with-dates"
    nlp = pipeline("ner", model=model_name, tokenizer=model_name,
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
                *dossiers, "NER", f"{nom_txt}_camembert-{torch.__version__}.json")
            path_output_bio = os.path.join(
                *dossiers, "NER", f"{nom_txt}_camembert-{torch.__version__}.bio")
            if os.path.exists(path_output) and not options.Force:
                print("Already DONE : ", path_output)
                continue

            texte = lire_fichier(path)
            entites = dico_resultats(texte, nlp)
            stocker(path_output, entites, is_json=True)

            entites_bio = bio_spacy(texte, nlp)
            with open(path_output_bio, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=' ', quotechar='|')
                writer.writerows(entites_bio)
