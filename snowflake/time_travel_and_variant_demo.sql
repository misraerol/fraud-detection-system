
USE WAREHOUSE FRAUD_WH;
USE DATABASE FRAUD_DB;
USE SCHEMA RAW;


SELECT COUNT(*) FROM TRANSACTIONS WHERE CLASS =1
DELETE FROM TRANSACTIONS WHERE CLASS =1

CREATE OR REPLACE TABLE TRANSACTIONS AS SELECT * FROM TRANSACTIONS AT (OFFSET => -120);


CREATE OR REPLACE TABLE TRANSACTION_METADATA (
    TRANSACTION_ID NUMBER AUTOINCREMENT,
    RAW_METADATA VARIANT
);

INSERT INTO TRANSACTION_METADATA (RAW_METADATA)
SELECT PARSE_JSON('{
  "merchant_category": "grocery",
  "device": {"type": "mobile", "os": "iOS"},
  "location": {"country": "CA", "city": "Toronto"}
}');

INSERT INTO TRANSACTION_METADATA (RAW_METADATA)
SELECT PARSE_JSON('{
  "merchant_category": "electronics",
  "device": {"type": "web"},
  "risk_flags": ["new_device", "high_amount"]
}');

SELECT 
    TRANSACTION_ID,
    RAW_METADATA:merchant_category::STRING as merchant_category,
    RAW_METADATA:device.type::STRING as device_type,
    RAW_METADATA:location.city::STRING as city,
    RAW_METADATA:risk_flags as risk_flags
FROM TRANSACTION_METADATA;

