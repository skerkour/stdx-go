package countries_test

import (
	"testing"

	"github.com/skerkour/stdx-go/countries"
)

func TestGetMap(t *testing.T) {
	// There are officialy 249 assigned country codes (https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
	// so with our user-defined code XX (for unknown) the total is 250
	expectedNumberOfCountries := 250

	countries := countries.All()

	if len(countries) != expectedNumberOfCountries {
		t.Errorf("Invalid number of countries. Got %d, expected: %d", len(countries), expectedNumberOfCountries)
	}
}

func TestGetCountry(t *testing.T) {
	tests := []struct {
		code string
		name string
	}{
		{"FR", "France"},
		{"XX", "Unknown"},
	}

	for _, test := range tests {
		countryName, _ := countries.Name(test.code)
		if countryName != test.name {
			t.Errorf("Code: %s -> Expected: %s | Got: %s", test.code, test.name, countryName)
		}
	}
}
