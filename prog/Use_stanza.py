from optparse import OptionParser
import re
import glob
from pathlib import Path
import stanza
import json
import os
import csv
import shutil
import warnings
warnings.simplefilter("ignore")
# TODO: gérer warnings
# from generic_tools import *


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


def load_stanza_model(lang: str = "fr") -> stanza.Pipeline:
    try:
        nlp = stanza.Pipeline(lang=lang, processors='tokenize,ner')
    except:
        stanza.download(lang=lang, logging_level='DEBUG')
        nlp = stanza.Pipeline(lang=lang, processors='tokenize,ner')
    return nlp


def lire_fichier(chemin, is_json=False):
    f = open(chemin, encoding='utf−8')
    if is_json == False:
        chaine = f.read()
    else:
        chaine = json.load(f)
    f.close()
    return chaine


def stocker(chemin, contenu, is_json=False, verbose=False):
    if verbose == True:
        print(f"  Output written in {chemin}")
    w = open(chemin, "w")
    if is_json == False:
        w.write(contenu)
    else:
        w.write(json.dumps(contenu, indent=2, ensure_ascii=False))
    w.close()


def get_ent_dict(ent) -> dict:
    """
    Args:
        ent : stanza entity

    Returns:
        dict: {label: PER, text: "Donald Trump", jalons: [0, 12]}
    """
    return {"label": ent.type, "text": ent.text, "jalons": [ent.start_char, ent.end_char]}


def dico_resultats(text, lang: str = "fr") -> dict[dict]:
    nlp = load_stanza_model(lang=lang)
    doc = nlp(text)
    return {f"entite_{i}": get_ent_dict(
        ent=ent) for i, ent in enumerate(doc.ents)}


def get_bio_tokens(doc) -> list[list]:
    print(doc.ents)
    return [
        [token.text, token.ner] for sentence in doc.sentences for token in sentence.tokens]


def bio_stanza(text: str, lang: str = "fr") -> list[str]:
    nlp = load_stanza_model(lang=lang)
    return get_bio_tokens(doc=nlp(text))


# print(dico_resultats(text="Donald Trump est un homme très innocent."))


if __name__ == "__main__":
    for lang in ["fr"]:
        liste_subcorpus = list(Path(path_corpora).glob("*"))
        print(liste_subcorpus)
        print(os.getcwd())
        if len(liste_subcorpus) == 0:
            print(
                f"Pas de dossier trouvé dans {path_corpora}, traitement terminé")
            exit()
        print(f"Using model: {lang}")
        for subcorpus in liste_subcorpus:
            print(f"Processing {subcorpus}")
            liste_txt = glob.glob(f"{subcorpus}/*_REF/*.txt")
            liste_txt += glob.glob(f"{subcorpus}/OCR/*/*.txt")
            print("  nombre de fichiers txt trouvés :", len(liste_txt))
            for path in liste_txt:
                dossiers = re.split("/", path)[:-1]
                nom_txt = re.split("/", path)[-1]
                path_ner = os.path.join(*dossiers, "NER")
                os.makedirs(path_ner, exist_ok=True)
                # format json
                path_output = f"{path_ner}/{nom_txt}_{lang}-{stanza.__version__}.json"
                print(path_output)
                # Pour le format bio
                path_output_bio = f"{path_ner}/{nom_txt}_{lang}-{stanza.__version__}.bio"
                print(path_output_bio)
                if os.path.exists(path_output) == True:
                    if options.Force == True:
                        print("  Recomputing :", path_output)
                    else:
                        print("Already DONE : ", path_output)
                        continue
                texte = lire_fichier(path)
                entites = dico_resultats(texte, lang=lang)
                stocker(path_output, entites, is_json=True)

                # Pour le format bio
                entites_bio = bio_stanza(text=texte, lang=lang)
                with open(path_output_bio, 'w', newline='') as file:
                    writer = csv.writer(file, delimiter=';', quotechar='|')
                    writer.writerows(entites_bio)
                    # writer.writerows([["Alice", 23], ["Bob", 27]])
            # Penser à comment lancer compute_distances
