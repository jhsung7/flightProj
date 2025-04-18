# FinalProject

## Flight Delay Prediction Considering Network Effects - Junghyeon Sung (Group 30)

- Parent paper's github link -> https://github.com/aravinda-1402/Flight_Delay_Prediction_using_Machine_Learning/tree/main

### 1. Dataset
- Flight Delay and Cancellation Dataset 2019-2023 (https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023?select=flights_sample_3m.csv)
  - Since the file size is too big to upload in Github, please download the file from the kaggle and run the code. Thank you!


### 2. Explanation of each column in the dataset
          FL_DATE: Date of the flight.
          AIRLINE: Name of the airline.
          AIRLINE_DOT: DOT identifier for the airline.
          AIRLINE_CODE: Code assigned to the airline.
          DOT_CODE: DOT identifier.
          FL_NUMBER: Flight number.
          ORIGIN: Origin airport code.
          ORIGIN_CITY: City of origin airport.
          DEST: Destination airport code.
          DEST_CITY: City of destination airport.
          CRS_DEP_TIME: Scheduled departure time.
          DEP_TIME: Actual departure time.
          DEP_DELAY: Departure delay.
          TAXI_OUT: Time spent taxiing out.
          WHEELS_OFF: Time when aircraft's wheels leave the ground.
          WHEELS_ON: Time when aircraft's wheels touch the ground.
          TAXI_IN: Time spent taxiing in.
          CRS_ARR_TIME: Scheduled arrival time.
          ARR_TIME: Actual arrival time.
          ARR_DELAY: Arrival delay.
          CANCELLED: Indicator if the flight was cancelled (1 for cancelled, 0 for not cancelled).
          CANCELLATION_CODE: Reason for cancellation (if applicable).
          DIVERTED: Indicator if the flight was diverted (1 for diverted, 0 for not diverted).
          CRS_ELAPSED_TIME: Scheduled elapsed time.
          ELAPSED_TIME: Actual elapsed time.
          AIR_TIME: Time spent in the air.
          DISTANCE: Distance traveled.
          DELAY_DUE_CARRIER: Delay due to carrier.
          DELAY_DUE_WEATHER: Delay due to weather.
          DELAY_DUE_NAS: Delay due to National Airspace System (NAS).
          DELAY_DUE_SECURITY: Delay due to security.
          DELAY_DUE_LATE_AIRCRAFT: Delay due to late aircraft arrival.
