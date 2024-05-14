import os,glob
liste_ocr=["kraken4.3.13.dev25","lectaurep-kraken4.3.13.dev25","tesseract0.3.10","REF"]
path_rep = "../ELTeC_OCRs/*/kraken4.3.13.dev25/*"
for ocr in liste_ocr:
    for rep in glob.glob(path_rep):

        path_output=rep.split("/")
        path_output="/".join([path_output[2],path_output[-1],path_output[-1]+"_"+ocr])
        print(path_output)
        os.mkdir("../archi/%s"%path_output)

