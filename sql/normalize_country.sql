-- ============================================================
-- COUNTRY NORMALIZATION
-- Purpose: Standardize inconsistent country values for reporting.
-- Original customers.country remains unchanged.
-- ============================================================

CREATE OR REPLACE VIEW customer_country_normalized AS

SELECT
    customer_id,
    first_name,
    last_name,

    CASE
        WHEN UPPER(TRIM(country)) IN
            ('INDIA', 'IN', 'BHARAT')
            THEN 'India'

        WHEN UPPER(TRIM(country)) IN
            ('UNITED STATES', 'USA', 'US', 'U.S.A')
            THEN 'United States'

        WHEN UPPER(TRIM(country)) IN
            ('UNITED KINGDOM', 'UK', 'U.K.', 'BRITAIN')
            THEN 'United Kingdom'

        WHEN UPPER(TRIM(country)) IN
            ('CANADA', 'CA')
            THEN 'Canada'

        WHEN UPPER(TRIM(country)) IN
            ('BRAZIL', 'BR')
            THEN 'Brazil'

        WHEN UPPER(TRIM(country)) IN
            ('GERMANY', 'DE')
            THEN 'Germany'

        WHEN UPPER(TRIM(country)) IN
            ('FRANCE', 'FR')
            THEN 'France'

        WHEN UPPER(TRIM(country)) IN
            ('MEXICO', 'MX')
            THEN 'Mexico'

        WHEN UPPER(TRIM(country)) IN
            ('JAPAN', 'JP')
            THEN 'Japan'

        WHEN UPPER(TRIM(country)) = 'PHILIPPINES'
            OR UPPER(TRIM(country)) = 'PH'
            THEN 'Philippines'

        WHEN UPPER(TRIM(country)) IN
            ('INDONESIA', 'ID')
            THEN 'Indonesia'

        WHEN UPPER(TRIM(country)) IN
            ('ITALY', 'IT')
            THEN 'Italy'

        WHEN UPPER(TRIM(country)) IN
            ('NIGERIA', 'NG')
            THEN 'Nigeria'

        WHEN UPPER(TRIM(country)) IN
            ('SOUTH AFRICA', 'ZA')
            THEN 'South Africa'

        WHEN UPPER(TRIM(country)) = 'SPAIN'
            OR UPPER(TRIM(country)) = 'ES'
            THEN 'Spain'

        WHEN UPPER(TRIM(country)) IN
            ('AUSTRALIA', 'AU')
            THEN 'Australia'

        WHEN UPPER(TRIM(country)) IN
            ('SOUTH KOREA', 'S. KOREA', 'KR')
            THEN 'South Korea'

        WHEN UPPER(TRIM(country)) IN
            ('ARGENTINA', 'AR')
            THEN 'Argentina'

        WHEN UPPER(TRIM(country)) IN
            ('POLAND', 'PL')
            THEN 'Poland'

        WHEN UPPER(TRIM(country)) IN
            ('UNITED ARAB EMIRATES', 'UAE', 'U.A.E')
            THEN 'United Arab Emirates'

        WHEN country IS NULL OR TRIM(country) = ''
            THEN 'Unknown'

        ELSE TRIM(country)
    END AS normalized_country

FROM customers;
