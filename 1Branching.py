# Conditional statements
# Branching - if-else statement

marks = 60

if (marks>=50):
    rslt='Pass' 
    msg = 'Congratulations !!'
    print(rslt+ ' - ' + msg)
##################  
marks = 40

if (marks>=50):
    rslt='Pass' 
    msg = 'Congratulations !!'
else:
    rslt='Fail'
    msg = 'Better luck next time'
print(rslt+ ' - ' + msg)


#################
#If Order Amount is greater than 2000 then 20% discount, else 10%

amount = 5000
disc_amt =0

if amount>=2000:
    disc=.2 
else:
    disc=.1
    
disc_amt =amount * disc
    
print(disc_amt)


###################
#If Order Amount is greater than 2000 then 20% discount, 
#else if greater than 1000 then 10% , else 5%

amount = 1200
disc_amt =0

if amount>=2000:
    disc=.2
elif amount >= 1500:
    disc=.15  
elif amount >= 1000:
    disc=.1 
else:
    disc=.05
    
disc_amt =amount * disc    

print(disc_amt)

#####################

#Nested if
amount = 1000
Ord_day = "Mon"
disc_amt =0
if (amount>=2000):
    if (Ord_day == "Mon"):
        disc = .2
    else:
        disc = .15   
else:
    disc=.1
    
disc_amt =amount * disc    

print(disc_amt)


######################################################








