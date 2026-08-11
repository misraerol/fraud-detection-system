USE WAREHOUSE FRAUD_WH;
USE DATABASE FRAUD_DB;
USE SCHEMA RAW;

SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'claude-3-5-sonnet',
    'Explain in two sentences what a fraud detection model does'
) AS response;