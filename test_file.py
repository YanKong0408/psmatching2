import psmatching2 as psm

path = "data-rhc.csv"
control="swang1"
cov = ["age", "sex", "race", "edu", "income", "ninsclas", "das2d3pc", "dnr1", "ca", "surv2md1", "aps1", "scoma1",
       "wtkilo1", "temp1", "meanbp1", "resp1", "hrt1", "pafi1", "paco21", "ph1", "wblc1", "hema1", "sod1", "pot1",
       "crea1", "bili1", "alb1", "resp", "card", "neuro", "gastr", "renal", "meta", "hema", "seps", "trauma", "ortho",
       "cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx", "liverhx", "gibledhx", "malighx", "immunhx",
       "transhx", "amihx"]
dep=["death"]

m=psm.PSMatch(path,control,cov,dep)
m.df.replace({"RHC":1,"No RHC":0},inplace=True)
m.calculate_proprnsity_scores()
m.match(caliper=0.2)
m.weighted_process(method="IPTW-P")
print(m.evaluate_dependent(m.df))
print(m.evaluate_dependent(m.matched_data))
print(m.evaluate_dependent(m.weighted_df))
m.evaluate_p_value(m.df)
m.evaluate_p_value(m.matched_data)
m.evaluate_p_value(m.weighted_df)
m.plot_matching_efficiency()
