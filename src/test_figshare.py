import urllib.request as u

URL = "https://springernature.figshare.com/ndownloader/files/51643487"   # metadonnees du match J03WPY
for name, headers in [("sans en-tete", {}), ("avec en-tete navigateur", {"User-Agent": "Mozilla/5.0"})]:
    try:
        r = u.urlopen(u.Request(URL, headers=headers), timeout=30)
        body = r.read()
        print("{:<26} statut {}  taille {} octets  debut : {!r}".format(name, r.status, len(body), body[:60]))
    except Exception as e:
        print("{:<26} erreur : {}".format(name, e))