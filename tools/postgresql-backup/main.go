package main

import (
	"context"
	"crypto/ecdh"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/skerkour/stdx-go/cron"
	"github.com/skerkour/stdx-go/yaml"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	configPath := flag.String("config", "", "path to config file")
	decryptFile := flag.String("decrypt", "", "path to encrypted file to decrypt")
	outputFile := flag.String("out", "", "output file (decrypt mode)")
	genKey := flag.Bool("generate", false, "generate a X25519 keypair")
	flag.Parse()

	if *genKey {
		generateKeypair()
		return
	}

	if *decryptFile != "" {
		runDecrypt(*decryptFile, *outputFile)
		return
	}

	if *configPath == "" {
		slog.Error("config flag is required")
		os.Exit(1)
	}

	cfg, err := loadConfig(*configPath)
	if err != nil {
		slog.Error("loading config", "error", err)
		os.Exit(1)
	}

	s3Client, err := createS3Client(cfg.S3)
	if err != nil {
		slog.Error("creating s3 client", "error", err)
		os.Exit(1)
	}

	scheduler := cron.New()

	for _, db := range cfg.Databases {
		if _, err := hex.DecodeString(db.PublicKey); err != nil {
			slog.Error("invalid public_key hex", "folder", db.Folder, "error", err)
			os.Exit(1)
		}

		_, err := scheduler.AddFunc(db.Cron, func() {
			slog.Info("running backup", "folder", db.Folder)
			if err := createBackup(context.Background(), s3Client, cfg.S3, db); err != nil {
				slog.Error("backup failed", "folder", db.Folder, "error", err)
			}
		})
		if err != nil {
			slog.Error("invalid cron expression", "folder", db.Folder, "cron", db.Cron, "error", err)
			os.Exit(1)
		}

		slog.Info("scheduled backup", "folder", db.Folder, "cron", db.Cron)
	}

	scheduler.Run()
}

func runDecrypt(inputPath, outputPath string) {
	keyHex := os.Getenv("KEY")
	if keyHex == "" {
		slog.Error("KEY environment variable is required")
		os.Exit(1)
	}

	if outputPath == "" {
		if strings.HasSuffix(inputPath, ".enc") {
			outputPath = strings.TrimSuffix(inputPath, ".enc")
		} else {
			slog.Error("-out is required when input file does not end with .enc")
			os.Exit(1)
		}
		outputPath = strings.TrimSuffix(outputPath, ".gz")
	}

	data, err := os.ReadFile(inputPath)
	if err != nil {
		slog.Error("reading input file", "error", err)
		os.Exit(1)
	}

	plaintext, err := decrypt(data, keyHex)
	if err != nil {
		slog.Error("decryption failed", "error", err)
		os.Exit(1)
	}

	decompressed, err := decompressGzip(plaintext)
	if err != nil {
		slog.Error("decompression failed", "error", err)
		os.Exit(1)
	}

	if err := os.WriteFile(outputPath, decompressed, 0600); err != nil {
		slog.Error("writing output file", "error", err)
		os.Exit(1)
	}

	slog.Info("decryption completed", "output", outputPath)
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	if cfg.S3.Endpoint == "" {
		return nil, errors.New("s3.endpoint is required")
	}
	if cfg.S3.Bucket == "" {
		return nil, errors.New("s3.bucket is required")
	}
	if cfg.S3.AccessKeyID == "" {
		return nil, errors.New("s3.access_key_id is required")
	}
	if cfg.S3.SecretAccessKey == "" {
		return nil, errors.New("s3.secret_access_key is required")
	}
	if cfg.S3.Region == "" {
		cfg.S3.Region = "auto"
	}

	for _, db := range cfg.Databases {
		if db.URL == "" {
			return nil, errors.New("database url is required")
		}
		if db.PublicKey == "" {
			return nil, errors.New("database public_key is required")
		}
		if len(db.PublicKey) != 64 {
			return nil, errors.New("database public_key is invalid")
		}

		if db.Cron == "" {
			return nil, errors.New("database cron is required")
		}
		if db.Folder == "" {
			return nil, errors.New("database folder is required")
		}
	}

	return &cfg, nil
}

func generateKeypair() {
	curve := ecdh.X25519()
	secretKey, err := curve.GenerateKey(nil)
	if err != nil {
		slog.Error("generating keypair", "error", err)
		os.Exit(1)
	}

	fmt.Printf("secret key: %s\n", hex.EncodeToString(secretKey.Bytes()))
	fmt.Printf("public key: %s\n", hex.EncodeToString(secretKey.PublicKey().Bytes()))
}
