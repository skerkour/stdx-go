package env

import (
	"errors"
	"net"
	"net/url"
	"testing"
	"time"
)

type BasicConfig struct {
	Host  string  `env:"HOST"`
	Port  int     `env:"PORT"`
	Debug bool    `env:"DEBUG"`
	Rate  float64 `env:"RATE"`
}

type TaglessConfig struct {
	Host string
	Port int
}

type RequiredConfig struct {
	Host string `env:"HOST,required"`
	Port int    `env:"PORT,default=8080"`
}

type NestedConfig struct {
	DB  DatabaseConfig `env:"DB"`
	App AppConfig
}

type DatabaseConfig struct {
	Host string `env:"HOST"`
	Port int    `env:"PORT"`
}

type AppConfig struct {
	Name string `env:"APP_NAME"`
}

type PtrFieldConfig struct {
	Host *string `env:"HOST"`
	Port *int    `env:"PORT"`
}

type SliceConfig struct {
	Hosts []string `env:"HOSTS"`
	Ports []int    `env:"PORTS"`
}

type TextUnmarshalerConfig struct {
	IP net.IP `env:"IP"`
}

type CustomType string

type CustomTypeConfig struct {
	Name CustomType `env:"NAME"`
}

type IgnoredFieldConfig struct {
	Host   string `env:"HOST"`
	Hidden string `env:"-"`
}

type DurationConfig struct {
	Timeout time.Duration `env:"TIMEOUT"`
}

func TestUnmarshalBasic(t *testing.T) {
	env := map[string]string{
		"HOST":  "localhost",
		"PORT":  "5432",
		"DEBUG": "true",
		"RATE":  "3.14",
	}

	var cfg BasicConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Host != "localhost" {
		t.Errorf("expected Host=localhost, got %q", cfg.Host)
	}
	if cfg.Port != 5432 {
		t.Errorf("expected Port=5432, got %d", cfg.Port)
	}
	if cfg.Debug != true {
		t.Errorf("expected Debug=true, got %v", cfg.Debug)
	}
	if cfg.Rate != 3.14 {
		t.Errorf("expected Rate=3.14, got %f", cfg.Rate)
	}
}

func TestUnmarshalNotPointer(t *testing.T) {
	var cfg BasicConfig
	err := Unmarshal(map[string]string{"HOST": "x"}, cfg)
	if err == nil {
		t.Fatal("expected error for non-pointer")
	}
}

func TestUnmarshalTagless(t *testing.T) {
	env := map[string]string{
		"Host": "localhost",
		"Port": "8080",
	}

	var cfg TaglessConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Host != "localhost" {
		t.Errorf("expected Host=localhost, got %q", cfg.Host)
	}
	if cfg.Port != 8080 {
		t.Errorf("expected Port=8080, got %d", cfg.Port)
	}
}

func TestUnmarshalRequired(t *testing.T) {
	var cfg RequiredConfig
	err := Unmarshal(map[string]string{}, &cfg)
	if err == nil {
		t.Fatal("expected error for missing required field")
	}
}

func TestUnmarshalDefault(t *testing.T) {
	env := map[string]string{
		"HOST": "localhost",
	}

	var cfg RequiredConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Host != "localhost" {
		t.Errorf("expected Host=localhost, got %q", cfg.Host)
	}
	if cfg.Port != 8080 {
		t.Errorf("expected Port=8080 (default), got %d", cfg.Port)
	}
}

func TestUnmarshalWithPrefix(t *testing.T) {
	env := map[string]string{
		"APP_HOST": "localhost",
		"APP_PORT": "8080",
	}

	type Config struct {
		Host string `env:"HOST"`
		Port int    `env:"PORT"`
	}

	var cfg Config
	err := UnmarshalWithPrefix("APP_", env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Host != "localhost" {
		t.Errorf("expected Host=localhost, got %q", cfg.Host)
	}
	if cfg.Port != 8080 {
		t.Errorf("expected Port=8080, got %d", cfg.Port)
	}
}

func TestUnmarshalNested(t *testing.T) {
	env := map[string]string{
		"DB_HOST":      "db.internal",
		"DB_PORT":      "3306",
		"App_APP_NAME": "myapp",
	}

	var cfg NestedConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.DB.Host != "db.internal" {
		t.Errorf("expected DB.Host=db.internal, got %q", cfg.DB.Host)
	}
	if cfg.DB.Port != 3306 {
		t.Errorf("expected DB.Port=3306, got %d", cfg.DB.Port)
	}
	if cfg.App.Name != "myapp" {
		t.Errorf("expected App.Name=myapp, got %q", cfg.App.Name)
	}
}

func TestUnmarshalNestedWithTag(t *testing.T) {
	type DBConfig struct {
		Host string `env:"HOST"`
	}

	type Config struct {
		DB DBConfig `env:"DB"`
	}

	env := map[string]string{
		"DB_HOST": "db.internal",
	}

	var cfg Config
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.DB.Host != "db.internal" {
		t.Errorf("expected DC.Host=db.internal, got %q", cfg.DB.Host)
	}
}

func TestUnmarshalNestedTagless(t *testing.T) {
	type DBConfig struct {
		Host string `env:"DB_HOST"`
	}

	type Config struct {
		DB DBConfig
	}

	env := map[string]string{
		"DB_DB_HOST": "db.internal",
	}

	var cfg Config
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.DB.Host != "db.internal" {
		t.Errorf("expected DB.Host=db.internal, got %q", cfg.DB.Host)
	}
}

func TestUnmarshalPtrField(t *testing.T) {
	env := map[string]string{
		"HOST": "localhost",
		"PORT": "8080",
	}

	var cfg PtrFieldConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Host == nil || *cfg.Host != "localhost" {
		t.Errorf("expected Host=localhost, got %v", *cfg.Host)
	}
	if cfg.Port == nil || *cfg.Port != 8080 {
		t.Errorf("expected Port=8080, got %d", *cfg.Port)
	}
}

func TestUnmarshalBoolVariants(t *testing.T) {
	tests := []struct {
		val  string
		want bool
	}{
		{"true", true}, {"TRUE", true}, {"True", true},
		{"1", true}, {"yes", true}, {"YES", true},
		{"false", false}, {"FALSE", false}, {"False", false},
		{"0", false}, {"no", false}, {"NO", false},
		{"on", true}, {"ON", true}, {"off", false}, {"OFF", false},
	}

	for _, tt := range tests {
		t.Run(tt.val, func(t *testing.T) {
			type Config struct {
				Val bool `env:"VAL"`
			}
			var cfg Config
			err := Unmarshal(map[string]string{"VAL": tt.val}, &cfg)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if cfg.Val != tt.want {
				t.Errorf("got %v, want %v", cfg.Val, tt.want)
			}
		})
	}
}

func TestUnmarshalIntTypes(t *testing.T) {
	type Config struct {
		A int   `env:"A"`
		B int8  `env:"B"`
		C int16 `env:"C"`
		D int32 `env:"D"`
		E int64 `env:"E"`
		F uint  `env:"F"`
		G uint8 `env:"G"`
	}

	env := map[string]string{
		"A": "-42", "B": "8", "C": "16", "D": "32", "E": "64",
		"F": "42", "G": "255",
	}

	var cfg Config
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.A != -42 || cfg.B != 8 || cfg.C != 16 || cfg.D != 32 || cfg.E != 64 {
		t.Fatal("int values mismatch")
	}
	if cfg.F != 42 || cfg.G != 255 {
		t.Fatal("uint values mismatch")
	}
}

func TestUnmarshalFloatTypes(t *testing.T) {
	type Config struct {
		F32 float32 `env:"F32"`
		F64 float64 `env:"F64"`
	}

	env := map[string]string{
		"F32": "3.14",
		"F64": "2.71828",
	}

	var cfg Config
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.F32 != 3.14 {
		t.Errorf("expected F32=3.14, got %f", cfg.F32)
	}
	if cfg.F64 != 2.71828 {
		t.Errorf("expected F64=2.71828, got %f", cfg.F64)
	}
}

func TestUnmarshalDuration(t *testing.T) {
	env := map[string]string{
		"TIMEOUT": "30s",
	}

	var cfg DurationConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Timeout != 30*time.Second {
		t.Errorf("expected 30s, got %v", cfg.Timeout)
	}
}

func TestUnmarshalSlices(t *testing.T) {
	env := map[string]string{
		"HOSTS": "a,b,c",
		"PORTS": "443,8080,3000",
	}

	var cfg SliceConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if len(cfg.Hosts) != 3 || cfg.Hosts[0] != "a" || cfg.Hosts[1] != "b" || cfg.Hosts[2] != "c" {
		t.Errorf("unexpected hosts: %v", cfg.Hosts)
	}
	if len(cfg.Ports) != 3 || cfg.Ports[0] != 443 || cfg.Ports[1] != 8080 || cfg.Ports[2] != 3000 {
		t.Errorf("unexpected ports: %v", cfg.Ports)
	}
}

func TestUnmarshalTextUnmarshaler(t *testing.T) {
	env := map[string]string{
		"IP": "192.168.1.1",
	}

	var cfg TextUnmarshalerConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if !cfg.IP.Equal(net.ParseIP("192.168.1.1")) {
		t.Errorf("expected IP 192.168.1.1, got %v", cfg.IP)
	}
}

func TestUnmarshalIgnoredField(t *testing.T) {
	env := map[string]string{
		"HOST": "localhost",
	}

	var cfg IgnoredFieldConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Host != "localhost" {
		t.Errorf("expected Host=localhost, got %q", cfg.Host)
	}
	if cfg.Hidden != "" {
		t.Errorf("expected Hidden=\"\", got %q", cfg.Hidden)
	}
}

func TestUnmarshalMissingFieldNoError(t *testing.T) {
	type Config struct {
		Host string  `env:"HOST"`
		Port int     `env:"PORT"`
		Rate float64 `env:"RATE"`
	}

	env := map[string]string{
		"HOST": "localhost",
	}

	var cfg Config
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Host != "localhost" {
		t.Errorf("expected Host=localhost, got %q", cfg.Host)
	}
	if cfg.Port != 0 {
		t.Errorf("expected Port=0 (zero), got %d", cfg.Port)
	}
	if cfg.Rate != 0 {
		t.Errorf("expected Rate=0 (zero), got %f", cfg.Rate)
	}
}

func TestUnmarshalWithPrefixNested(t *testing.T) {
	type AppConfig struct {
		DBHost string `env:"DB_HOST"`
		DBPort int    `env:"DB_PORT"`
	}

	env := map[string]string{
		"MYAPP_DB_HOST": "db.internal",
		"MYAPP_DB_PORT": "5432",
	}

	var cfg AppConfig
	err := UnmarshalWithPrefix("MYAPP_", env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.DBHost != "db.internal" {
		t.Errorf("expected DBHost=db.internal, got %q", cfg.DBHost)
	}
	if cfg.DBPort != 5432 {
		t.Errorf("expected DBPort=5432, got %d", cfg.DBPort)
	}
}

func TestUnmarshalCustomType(t *testing.T) {
	env := map[string]string{
		"NAME": "foobar",
	}

	var cfg CustomTypeConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if string(cfg.Name) != "foobar" {
		t.Errorf("expected Name=foobar, got %q", cfg.Name)
	}
}

func TestUnmarshalEmptyMap(t *testing.T) {
	type Config struct {
		Host string `env:"HOST"`
		Port int    `env:"PORT"`
	}

	var cfg Config
	err := Unmarshal(map[string]string{}, &cfg)
	if err != nil {
		t.Fatal(err)
	}
}

func TestUnmarshalInvalidInt(t *testing.T) {
	type Config struct {
		Port int `env:"PORT"`
	}

	var cfg Config
	err := Unmarshal(map[string]string{"PORT": "notanint"}, &cfg)
	if err == nil {
		t.Fatal("expected error for invalid int")
	}
}

func TestUnmarshalInvalidBool(t *testing.T) {
	type Config struct {
		Debug bool `env:"DEBUG"`
	}

	var cfg Config
	err := Unmarshal(map[string]string{"DEBUG": "maybe"}, &cfg)
	if err == nil {
		t.Fatal("expected error for invalid bool")
	}
}

func TestUnmarshalInvalidFloat(t *testing.T) {
	type Config struct {
		Rate float64 `env:"RATE"`
	}

	var cfg Config
	err := Unmarshal(map[string]string{"RATE": "notafloat"}, &cfg)
	if err == nil {
		t.Fatal("expected error for invalid float")
	}
}

func TestUnmarshalErrorImplements(t *testing.T) {
	var cfg BasicConfig
	err := Unmarshal(map[string]string{}, &cfg)

	// Check the error type can be inspected
	var target *strconvErr
	if err != nil {
		if errors.As(err, &target) {
			t.Fatalf("unexpected strconvErr: %v", target)
		}
	}
}

func TestUnmarshalWithPrefixEmpty(t *testing.T) {
	type Config struct {
		Host string `env:"HOST"`
	}

	env := map[string]string{
		"HOST": "should-not-match",
	}

	var cfg Config
	err := UnmarshalWithPrefix("APP_", env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Host != "" {
		t.Errorf("expected empty Host (prefix mismatch), got %q", cfg.Host)
	}
}

type ByteSliceConfig struct {
	Data []byte `env:"DATA"`
}

func TestUnmarshalByteSlice(t *testing.T) {
	env := map[string]string{
		"DATA": "SGVsbG8gV29ybGQ=",
	}

	var cfg ByteSliceConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	want := []byte("Hello World")
	if string(cfg.Data) != string(want) {
		t.Errorf("expected %q, got %q", want, cfg.Data)
	}
}

func TestUnmarshalByteSliceEmpty(t *testing.T) {
	env := map[string]string{
		"DATA": "",
	}

	var cfg ByteSliceConfig
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if len(cfg.Data) != 0 {
		t.Errorf("expected empty slice, got %v", cfg.Data)
	}
}

func TestUnmarshalByteSliceInvalidBase64(t *testing.T) {
	env := map[string]string{
		"DATA": "not-valid-base64!!!",
	}

	var cfg ByteSliceConfig
	err := Unmarshal(env, &cfg)
	if err == nil {
		t.Fatal("expected error for invalid base64")
	}
}

func TestUnmarshalByteSliceMissing(t *testing.T) {
	var cfg ByteSliceConfig
	err := Unmarshal(map[string]string{}, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Data != nil {
		t.Errorf("expected nil, got %v", cfg.Data)
	}
}

func TestUnmarshalByteSliceWithPrefix(t *testing.T) {
	type Config struct {
		Key []byte `env:"KEY"`
	}

	env := map[string]string{
		"APP_KEY": "dGVzdC1rZXk=",
	}

	var cfg Config
	err := UnmarshalWithPrefix("APP_", env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	want := []byte("test-key")
	if string(cfg.Key) != string(want) {
		t.Errorf("expected %q, got %q", want, cfg.Key)
	}
}

func TestUnmarshalURL(t *testing.T) {
	type Config struct {
		URL url.URL `env:"URL,required"`
	}

	env := map[string]string{
		"URL": "postgres://user:pass@localhost:5432/db?sslmode=disable",
	}

	var cfg Config
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.URL.Scheme != "postgres" {
		t.Errorf("expected scheme=postgres, got %q", cfg.URL.Scheme)
	}
	if cfg.URL.Host != "localhost:5432" {
		t.Errorf("expected host=localhost:5432, got %q", cfg.URL.Host)
	}
	if cfg.URL.Path != "/db" {
		t.Errorf("expected path=/db, got %q", cfg.URL.Path)
	}
}

func TestUnmarshalURLWithSubnested(t *testing.T) {
	type Database struct {
		URL      url.URL `env:"URL,required"`
		PoolSize uint32  `env:"POOL_SIZE,default=85"`
	}

	type Config struct {
		Port     uint16   `env:"PORT,default=443"`
		Database Database `env:"DATABASE"`
	}

	env := map[string]string{
		"DATABASE_URL":       "mysql://user:pass@db.internal:3306/app",
		"DATABASE_POOL_SIZE": "42",
	}

	var cfg Config
	err := Unmarshal(env, &cfg)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Port != 443 {
		t.Errorf("expected Port=443 (default), got %d", cfg.Port)
	}
	if cfg.Database.URL.Scheme != "mysql" {
		t.Errorf("expected scheme=mysql, got %q", cfg.Database.URL.Scheme)
	}
	if cfg.Database.URL.Host != "db.internal:3306" {
		t.Errorf("expected host=db.internal:3306, got %q", cfg.Database.URL.Host)
	}
	if cfg.Database.URL.Path != "/app" {
		t.Errorf("expected path=/app, got %q", cfg.Database.URL.Path)
	}
	if cfg.Database.PoolSize != 42 {
		t.Errorf("expected PoolSize=42, got %d", cfg.Database.PoolSize)
	}
}

func TestUnmarshalURLMissing(t *testing.T) {
	type Config struct {
		URL url.URL `env:"URL,required"`
	}

	var cfg Config
	err := Unmarshal(map[string]string{}, &cfg)
	if err == nil {
		t.Fatal("expected error for missing URL")
	}
}
