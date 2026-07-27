-- Part 1.2: SQL-defined Unity Catalog function.

CREATE OR REPLACE FUNCTION cs4603.default.to_billions(
    amount DOUBLE
)
RETURNS DOUBLE
COMMENT 'Convert an amount expressed in base units into billions.'
RETURN amount / 1000000000.0;


-- Direct verification required by the assignment.

SELECT cs4603.default.to_billions(2400000000.0)
       AS amount_in_billions;


-- Inspect all four functions.

SHOW FUNCTIONS IN cs4603.default LIKE '*growth_rate*';
SHOW FUNCTIONS IN cs4603.default LIKE '*percentage_change*';
SHOW FUNCTIONS IN cs4603.default LIKE '*compare_values*';
SHOW FUNCTIONS IN cs4603.default LIKE '*to_billions*';


DESCRIBE FUNCTION EXTENDED cs4603.default.growth_rate;
DESCRIBE FUNCTION EXTENDED cs4603.default.percentage_change;
DESCRIBE FUNCTION EXTENDED cs4603.default.compare_values;
DESCRIBE FUNCTION EXTENDED cs4603.default.to_billions;


-- Governance evidence.

GRANT EXECUTE ON FUNCTION cs4603.default.growth_rate
TO `27100380@lums.edu.pk`;

GRANT EXECUTE ON FUNCTION cs4603.default.percentage_change
TO `27100380@lums.edu.pk`;

GRANT EXECUTE ON FUNCTION cs4603.default.compare_values
TO `27100380@lums.edu.pk`;

GRANT EXECUTE ON FUNCTION cs4603.default.to_billions
TO `27100380@lums.edu.pk`;


SHOW GRANTS ON FUNCTION cs4603.default.growth_rate;
SHOW GRANTS ON FUNCTION cs4603.default.percentage_change;
SHOW GRANTS ON FUNCTION cs4603.default.compare_values;
SHOW GRANTS ON FUNCTION cs4603.default.to_billions;