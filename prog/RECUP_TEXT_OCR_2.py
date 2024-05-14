# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import glob, time

def lire_fichier (chemin):
    f = open(chemin , encoding = 'utf−8')
    chaine = f.read ()
    f.close ()
    return chaine

def nom_fichier(nom_fichier):
    nom_fichier= nom_fichier.split("/")
    nom_fichier="_".join([nom_fichier[4],nom_fichier[3]])
    return nom_fichier

def stocker( chemin, contenu):

    w =open(chemin, "w")
    w.write(contenu )
    w.close()
#    print(chemin)
start = time.time()
liste_texte=[]


liste_ocr=["kraken4.3.13.dev25","lectaurep-kraken4.3.13.dev25","tesseract0.3.10"]
n=5
for ocr in liste_ocr:
    path_corpora = "../ELTeC_OCRs/salve%s/%s/*"%(n,ocr)

    for subcorpus in glob.glob("%s/"%path_corpora):

        liste_texte=[]
        for fichiertext in sorted(glob.glob("%s/*"%subcorpus)):
            #print(fichiertext)
            path_output = nom_fichier(fichiertext)
            print(path_output)

            chaine=lire_fichier(fichiertext)

            liste_texte.append(chaine)

            texte= "\n".join(liste_texte)
    #print(texte)


            stocker("../OUTPUT_TEXT_recup/salve%s/%s.txt"%(n,path_output),texte)
end = time.time()
print(format(end-start))    
    

        
        
 
       