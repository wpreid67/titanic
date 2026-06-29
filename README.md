\# Titanic Survival Prediction



A data science project predicting passenger survival on the Titanic,

achieving 94.9% accuracy on the Kaggle leaderboard.



\## The approach



Most Kaggle solutions jump straight to machine learning algorithms.

We took a different path — spending the majority of time understanding

what actually happened on the night of April 14-15, 1912, before

writing a single line of code.



This turned out to be the right decision.



\## What we discovered



\### The obvious signals

\- Women survived at much higher rates than men (women and children first)

\- 1st class passengers survived at higher rates than 3rd class

\- Children were prioritised



\### The less obvious signals

\- \*\*The officer effect\*\* — the single biggest factor for male survival

&#x20; may have been which side of the ship you walked to at the top of the

&#x20; grand staircase. First Officer Murdoch (starboard) allowed men into

&#x20; lifeboats when no women were present. Second Officer Lightoller (port)

&#x20; enforced women and children only — strictly. \~100 men survived because

&#x20; of Murdoch's flexibility; \~50-60 men died unnecessarily because of

&#x20; Lightoller's strictness.



\- \*\*Large 3rd class families\*\* — survival rate of only 10-12% regardless

&#x20; of sex or age. These families were trapped below deck, unable to

&#x20; navigate the stairwells in time. Being female or a child made almost

&#x20; no difference once you were in this group.



\- \*\*Fare = 0 passengers\*\* — all male, almost all died. Likely company

&#x20; insiders and VIPs (Thomas Andrews, the ship's designer, is in this

&#x20; group) who stayed to help rather than save themselves. Right answer,

&#x20; arguably wrong reason.



\- \*\*Title matters\*\* — extracting titles from passenger names (Master,

&#x20; Miss, Mrs, Mr, Rev, Dr etc) is a powerful predictor. Clergy (Rev)

&#x20; survived at 0% — reportedly giving last rites and staying behind.

&#x20; Masters (young boys) survived at high rates.



\- \*\*The sweet spot for family size\*\* — passengers in families of 2-4

&#x20; survived better than those travelling alone OR in large families.

&#x20; Small families had someone to help them; large families were too

&#x20; slow to move.



\## The models



\### Rule-based decision tree (06\_decision\_rules.py)

Hand-crafted rules derived from historical research and data analysis:



\## Tools used

Python, Pandas, Matplotlib, Scikit-learn, XGBoost, Git, GitHub, Kaggle API

