from svc_model import pred_clean

string = str(input("Digite seu Feedback:"))

classe, proba0, proba1 = pred_clean(string)

print(f'A classe do seu feedback é: {classe}')
print(f'A probabilidade do seu feedback ser negativo é: {proba0}%')
print(f'A probabilidade do seu feedback ser positivo é: {proba1}%')