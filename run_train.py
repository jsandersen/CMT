# Run one or multiple training scripts 

# TEMPLATE
#
# from training.{IMDB|NewsGroups|Toxic}_{CNN1|CNN2|BERT}_{BL|BBB|MCD|EN} import build
# build()


from training.IMDB_CNN1_BL import build
build()

