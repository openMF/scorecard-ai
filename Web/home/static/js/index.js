// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// AI-Assisted Development Notice:
// This file was extracted from inline <script> in index.html with the
// assistance of GitHub Copilot. Core logic is preserved from the original;
// structural improvements (IIFE scope, null-checks, createElement) were
// reviewed and validated by the contributor.

(() => {
    const dateFeaturesNode = document.getElementById('date-features-data');
    const dateFeatures = dateFeaturesNode ? JSON.parse(dateFeaturesNode.textContent) : [];
    const DATE_FEATURES = new Set(dateFeatures);

    const featureForm = document.getElementById('feature-form');
    const selectedFeaturesContainer = document.getElementById('selected-features-container');
    const column1 = document.getElementById('column1');
    const column2 = document.getElementById('column2');
    const selectAllCheckbox = document.getElementById('select-all');

    if (!featureForm || !selectedFeaturesContainer || !column1 || !column2 || !selectAllCheckbox) {
        return;
    }

    const featureCheckboxes = featureForm.querySelectorAll('input[type="checkbox"][name="selected_features"]');

    featureForm.addEventListener('change', (event) => {
        if (event.target.type === 'checkbox' && event.target.name === 'selected_features') {
            const feature = event.target.value;
            if (event.target.checked) {
                addFeatureInput(feature);
            } else {
                removeFeatureInput(feature);
            }
            updateSelectAllCheckbox();
        }
    });

    selectedFeaturesContainer.addEventListener('click', (event) => {
        if (!event.target.classList.contains('remove-btn')) {
            return;
        }
        const featureInput = event.target.closest('.feature-input');
        if (!featureInput) {
            return;
        }
        const feature = featureInput.dataset.feature;
        removeFeatureInput(feature);
        uncheckFeatureCheckbox(feature);
        updateSelectAllCheckbox();
    });

    selectAllCheckbox.addEventListener('change', () => {
        const isChecked = selectAllCheckbox.checked;
        featureCheckboxes.forEach((checkbox) => {
            checkbox.checked = isChecked;
            if (isChecked) {
                addFeatureInput(checkbox.value);
            } else {
                removeFeatureInput(checkbox.value);
            }
        });
    });

    function naturalSorter(as, bs) {
        let a;
        let b;
        let a1;
        let b1;
        let i = 0;
        const rx = /(\d+)|(\D+)/g;
        const rd = /\d/;
        if (isFinite(as) && isFinite(bs)) return as - bs;
        a = String(as).toLowerCase();
        b = String(bs).toLowerCase();
        if (a === b) return 0;
        while ((a1 = a.match(rx)) && (b1 = b.match(rx))) {
            if ((a1 = a1[i]) === (b1 = b1[i])) {
                i++;
                continue;
            }
            return rd.test(a1) && rd.test(b1) ? a1 - b1 : a1 > b1 ? 1 : -1;
        }
        return a > b ? 1 : -1;
    }

    function addFeatureInput(feature) {
        if (selectedFeaturesContainer.querySelector(`.feature-input[data-feature="${feature}"]`)) {
            return;
        }

        const isDate = DATE_FEATURES.has(feature);
        const inputType = isDate ? 'date' : 'number';
        const placeholder = isDate ? '' : 'Enter a number';
        const hint = isDate ? '<span class="feature-hint">YYYY-MM-DD or numeric days</span>' : '';

        const featureInput = document.createElement('div');
        featureInput.className = 'feature-input';
        featureInput.dataset.feature = feature;

        const input = document.createElement('input');
        input.type = inputType;
        input.id = `${feature}-input`;
        input.name = feature;
        input.placeholder = placeholder;
        if (!isDate) {
            input.step = 'any';
        }

        const label = document.createElement('label');
        label.setAttribute('for', `${feature}-input`);
        label.innerHTML = `${feature} ${hint}`;

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'remove-btn';
        removeBtn.textContent = 'Remove';

        featureInput.appendChild(label);
        featureInput.appendChild(input);
        featureInput.appendChild(removeBtn);

        const columnToAddTo = column1.children.length <= column2.children.length ? column1 : column2;
        columnToAddTo.appendChild(featureInput);

        sortFeatureInputs(column1);
        sortFeatureInputs(column2);
    }

    function removeFeatureInput(feature) {
        const featureInput = selectedFeaturesContainer.querySelector(`.feature-input[data-feature="${feature}"]`);
        if (featureInput) {
            featureInput.remove();
        }
    }

    function uncheckFeatureCheckbox(feature) {
        const checkbox = featureForm.querySelector(`input[type="checkbox"][value="${feature}"]`);
        if (checkbox) {
            checkbox.checked = false;
        }
    }

    function sortFeatureInputs(column) {
        const inputs = Array.from(column.children);
        inputs.sort((a, b) => naturalSorter(a.dataset.feature, b.dataset.feature));
        column.innerHTML = '';
        inputs.forEach((input) => column.appendChild(input));
    }

    function updateSelectAllCheckbox() {
        const allChecked = Array.from(featureCheckboxes).every((checkbox) => checkbox.checked);
        selectAllCheckbox.checked = allChecked;
    }

    function fillSpecificValues() {
        const selectedFeatureSet = document.getElementById('feature_set').value;

        const specificValuesF1 = {
            activation_date: '2020-09-01',
            gender_cv_id: 2719,
            date_of_birth: '1973-09-01',
            legal_form_enum: 1,
            principal_amount: 15586.60,
            interest_period_frequency_enum: 2,
            interest_method_enum: 1,
            interest_calculated_in_period_enum: 0,
            approvedon_date: '2020-08-01',
            expected_disbursedon_date: '2020-08-01',
            disbursedon_date: '2020-08-02',
            expected_maturedon_date: '2022-07-01',
            maturedon_date: '2022-02-15',
            transaction_type_enum: 2,
            transaction_date: '2021-04-15',
            amount: 4381.82,
            submitted_on_date: '2023-08-15',
        };

        const specificValuesF2 = {
            activation_date: '2020-06-01',
            gender_cv_id: 16,
            date_of_birth: '1974-06-15',
            validatedon_date: '2020-06-01',
            legal_form_enum: 1,
            principal_amount: 10000,
            nominal_interest_rate_per_period: 0,
            interest_period_frequency_enum: 2,
            annual_nominal_interest_rate: 0,
            interest_method_enum: 1,
            interest_calculated_in_period_enum: 0,
            term_frequency: 10,
            number_of_repayments: 10,
            approvedon_date: '2022-09-01',
            expected_disbursedon_date: '2022-09-01',
            disbursedon_date: '2022-09-01',
            expected_maturedon_date: '2023-07-15',
            maturedon_date: '2023-07-15',
            transaction_type_enum: 2,
            transaction_date: '2023-02-01',
            amount: 1000,
            submitted_on_date: '2023-08-01',
            created_date: '2023-08-01',
        };

        const specificValues = selectedFeatureSet === 'F1' ? specificValuesF1 : specificValuesF2;

        const inputs = document.querySelectorAll('.feature-input input');
        inputs.forEach((input) => {
            const featureName = input.name;
            if (specificValues.hasOwnProperty(featureName)) {
                input.value = specificValues[featureName];
            }
        });
    }

    featureCheckboxes.forEach((checkbox) => {
        if (checkbox.checked) {
            addFeatureInput(checkbox.value);
        }
    });

    updateSelectAllCheckbox();

    window.fillSpecificValues = fillSpecificValues;
})();
