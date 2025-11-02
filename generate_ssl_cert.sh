#!/bin/bash
# Generate self-signed SSL certificate for HTTPS

echo "Generating self-signed SSL certificate for HTTPS..."
echo ""

# Create certs directory
mkdir -p certs

# Generate private key and certificate
openssl req -x509 -newkey rsa:4096 -nodes \
    -out certs/cert.pem \
    -keyout certs/key.pem \
    -days 365 \
    -subj "/C=US/ST=State/L=City/O=Restaurant Booking/CN=localhost"

echo ""
echo "✓ SSL certificate generated:"
echo "  Certificate: certs/cert.pem"
echo "  Private Key: certs/key.pem"
echo ""
echo "To start HTTPS server:"
echo "  python api_server.py --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem"
echo ""
echo "Note: This is a self-signed certificate for development."
echo "For production, use a certificate from a trusted CA (Let's Encrypt, etc.)"
