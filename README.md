## AV Club Mahindra DataOlympics

### Problem Statement
Food & Beverages Spend Prediction in Club Mahindra Resorts

Club Mahindra (Club M) makes significant revenue from Food and Beverages (F&B) sales in their resorts. The members of Club M are offered a wide variety of items in either buffet or À la carte form. Following are some benefits that the model to predict the spend by a member in their next visit to a resort will bring:

- Predicting the F&B spend of a member in a resort would help in improving the pre-sales during resort booking through web and mobile app
- Targeted campaigns to suit the member taste and preference of F&B
- Providing members in the resort with a customized experience and offers
- Help resort kitchen to plan the inventory and food quantity to be prepared in advance

Given the information related to resort, club member, reservation etc. the task is to predict average spend per room night on food and beverages for the each reservation in the test set.

### Scoring Metric

Score = 100*RMSE

### Solution
The final solution was weighted average of 2 XGB models

- Public Score : 96.64
- Private Score : 97.8

### Things I could have added
- Extra features on member level information (like- # of restaurants visited, average spending pattern per customer)
- holidays

Competition Link: https://datahack.analyticsvidhya.com/contest/club-mahindra-dataolympics/
