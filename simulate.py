from ONDOC import OnDoc
from OnDoc_plus import OnDoc_plus
from COFE import COFE
from COFE_plus import COFE_plus

# ondoc = OnDoc()
# ondoc.receive_dag()
# ondoc.schedule()
# str = ondoc.str()
# print(str)

# cofe = COFE()
# cofe.receive_dag()
# cofe.schedule()
# str = cofe.str()
# print(str) 

# cofe_plus = COFE_plus()
# cofe_plus.receive_dag()
# cofe_plus.schedule()
# str = cofe_plus.str()
# print(str)

mine = OnDoc_plus()
mine.receive_dag()
mine.schedule()
str = mine.str()
print(str)