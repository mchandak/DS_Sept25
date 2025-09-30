#For loop
ord_amount = [4000,1000,2200,800]

for val in ord_amount:
    print("Order Amount :" ,val)

######

#For and branching
ord_amount = [4000,1000,2200,800]
for val in ord_amount:    
    if val>=2000:
        disc=.2
    else:
        disc=.1
    
    disc_amt = val * disc    
    print("Order Amount :", val)
    print("Discount Amount :", disc_amt)
    
#######

# Using index 
#n= len(ord_amount)

for i in range(0,4):
    print("Order Amount :" ,ord_amount[i])
    print("Index:" , i)
    
  
#######################################

#While loop
ord_amount = [2000,1000,2200,800,1200,2100,2500]

x=0
tot = 0
while(x <3):
    tot = tot + ord_amount[x]
    print (x, "print1 :", ord_amount[x])
    print (x,"print2 :", tot)
    x = x+1
  

######################################
#Defining function (user defined function -udf)

a = 10                  
b = 20
c= a+b
c

def myfunction():
    a = 10                  
    b = 20
    c= a+b
    return c
    
myfunction()

def myfunction2(a,b):
    c= a+b
    return c

myfunction2(50,20)

def ExamResult(marks):
    if (marks>=50):
        rslt='Pass' 
        msg = 'Congratulations !!'
    else:
        rslt='Fail'
        msg = 'Better luck next time'
    output = rslt+ ' - ' + msg
    return(output)

# Calling function
mark=40
ExamResult(mark)

#Functions and for loop

lstMarks =[60,20,40,70,55]

for val in lstMarks:
    rslt1 = ExamResult(val)     #Calling function
    print(val, rslt1)

#Built-in functions
ord_amount = [4000,1000,2200,800]
sum(ord_amount)
max(ord_amount)
min(ord_amount)

pow(4,2)
4**2

round(40.58)
round(40.56,1)
round(40.877333,0)






