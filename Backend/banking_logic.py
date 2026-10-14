# SIO Voice-First Banking Assistant - Module 4: Core Banking Engine & Database    # File title comment
# This script manages mock banking operations, user accounts, and transactions.  # Module description comment
# Supports all 7 banking services: Balance, Transfers, Passbook, Bills, Recharge # Service capabilities comment
# ATM Locator, and Card Blocking / Help. Shows Balance BEFORE and AFTER actions. # Ledger transparency comment

import os  # Import os module for managing database directory and file paths
import sys  # Import sys module for standard runtime configuration
import sqlite3  # Import sqlite3 module to manage lightweight local relational database
import datetime  # Import datetime module to generate timestamps for bank ledger records
import random  # Import random module to generate realistic transaction reference IDs
from typing import Dict, Any, List, Optional  # Import type hint annotations for clean signatures

# Reconfigure standard output streams to UTF-8 for displaying Hindi and symbols   # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if stdout supports stream reconfiguration
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout encoding to UTF-8
if hasattr(sys.stderr, 'reconfigure'):  # Check if stderr supports stream reconfiguration
    sys.stderr.reconfigure(encoding='utf-8')  # Set stderr encoding to UTF-8

# Determine path for the persistent SQLite database file                         # Description comment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get project root folder path
DATA_DIR = os.path.join(BASE_DIR, "data")  # Path to data directory for database storage
os.makedirs(DATA_DIR, exist_ok=True)  # Create data directory if it does not already exist
DB_PATH = os.path.join(DATA_DIR, "sio_banking.db")  # Full path to the SQLite banking database file

# Default primary user account identifier used across SIO sessions                # Description comment
DEFAULT_ACCOUNT_NUM = "SB-8839201452"  # Standard mock savings bank account number

# Function to get an active SQLite database connection with row dictionary access # Description comment
def get_db_connection():  # Define database connection factory function
    conn = sqlite3.connect(DB_PATH)  # Connect to local SQLite database file
    conn.row_factory = sqlite3.Row  # Enable row factory so columns can be accessed by column name
    return conn  # Return the configured database connection object

# Function to initialize database tables and seed comprehensive dummy data       # Description comment
def init_banking_database():  # Define function to bootstrap banking tables and dummy records
    conn = get_db_connection()  # Open database connection
    cursor = conn.cursor()  # Create database cursor for executing DDL and DML queries
    # Table 1: User bank accounts storing balance and card status
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        account_number TEXT PRIMARY KEY,
        holder_name TEXT NOT NULL,
        phone_number TEXT NOT NULL,
        account_type TEXT NOT NULL,
        balance REAL NOT NULL,
        card_number TEXT NOT NULL,
        card_status TEXT NOT NULL
    )
    """)  # Create accounts table if it does not exist
    # Table 2: Transaction history ledger recording every financial movement
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        txn_id TEXT PRIMARY KEY,
        account_number TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        type TEXT NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        details TEXT NOT NULL,
        balance_before REAL NOT NULL,
        balance_after REAL NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (account_number) REFERENCES accounts(account_number)
    )
    """)  # Create transactions table if it does not exist
    # Table 3: Registered beneficiaries for quick funds transfer
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS beneficiaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_number TEXT NOT NULL,
        name TEXT NOT NULL,
        nick_name TEXT NOT NULL,
        beneficiary_acc TEXT NOT NULL,
        bank_name TEXT NOT NULL
    )
    """)  # Create beneficiaries table if it does not exist
    # Table 4: Utility bills due for payments
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS utility_bills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_number TEXT NOT NULL,
        bill_type TEXT NOT NULL,
        provider_name TEXT NOT NULL,
        consumer_id TEXT NOT NULL,
        amount REAL NOT NULL,
        due_date TEXT NOT NULL,
        status TEXT NOT NULL
    )
    """)  # Create utility bills table if it does not exist
    # Table 5: Nearby ATM and Branch locations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS atm_locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        address TEXT NOT NULL,
        distance_meters INTEGER NOT NULL,
        cash_status TEXT NOT NULL,
        wheelchair_access TEXT NOT NULL
    )
    """)  # Create ATM locations table if it does not exist
    # Seed default account record if accounts table is empty
    cursor.execute("SELECT COUNT(*) FROM accounts WHERE account_number = ?", (DEFAULT_ACCOUNT_NUM,))  # Check existing
    if cursor.fetchone()[0] == 0:  # If default user account not found
        cursor.execute("""
        INSERT INTO accounts (account_number, holder_name, phone_number, account_type, balance, card_number, card_status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (DEFAULT_ACCOUNT_NUM, "Shubham Gala", "9876543210", "Savings Account", 12500.0, "XXXX-XXXX-XXXX-4129", "ACTIVE"))  # Seed account
    # Seed default beneficiaries if empty
    cursor.execute("SELECT COUNT(*) FROM beneficiaries WHERE account_number = ?", (DEFAULT_ACCOUNT_NUM,))  # Check beneficiaries
    if cursor.fetchone()[0] == 0:  # If no beneficiaries exist
        sample_beneficiaries = [  # List of realistic sample beneficiaries
            (DEFAULT_ACCOUNT_NUM, "Raju Sharma", "Raju", "SB-1092837461", "State Bank of India"),  # Raju
            (DEFAULT_ACCOUNT_NUM, "Amit Kumar", "Amit", "HDFC-9920148201", "HDFC Bank"),  # Amit
            (DEFAULT_ACCOUNT_NUM, "Ramesh Patil", "Ramesh", "BOI-7719283019", "Bank of India"),  # Ramesh
            (DEFAULT_ACCOUNT_NUM, "Priya Nair", "Priya", "ICIC-4401928301", "ICICI Bank"),  # Priya
        ]  # End of sample beneficiaries
        cursor.executemany("INSERT INTO beneficiaries (account_number, name, nick_name, beneficiary_acc, bank_name) VALUES (?, ?, ?, ?, ?)", sample_beneficiaries)  # Insert
    # Seed initial transaction history if empty
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE account_number = ?", (DEFAULT_ACCOUNT_NUM,))  # Check history
    if cursor.fetchone()[0] == 0:  # If no transactions exist
        sample_history = [  # Initial mock passbook entries showing past activity
            ("TXN892011", DEFAULT_ACCOUNT_NUM, "2026-10-05 10:30:00", "CREDIT", "Salary", 25000.0, "Monthly Pension / Salary Credit", 0.0, 25000.0, "SUCCESS"),  # Credit
            ("TXN892012", DEFAULT_ACCOUNT_NUM, "2026-10-06 14:15:00", "DEBIT", "Bill Payment", 1200.0, "Electricity Bill (MSEB)", 25000.0, 23800.0, "SUCCESS"),  # Bill
            ("TXN892013", DEFAULT_ACCOUNT_NUM, "2026-10-08 17:40:00", "DEBIT", "Recharge", 299.0, "Mobile Prepaid Recharge (Jio)", 23800.0, 23501.0, "SUCCESS"),  # Recharge
            ("TXN892014", DEFAULT_ACCOUNT_NUM, "2026-10-10 11:20:00", "DEBIT", "ATM Cash", 5000.0, "ATM Cash Withdrawal", 23501.0, 18501.0, "SUCCESS"),  # Cash
            ("TXN892015", DEFAULT_ACCOUNT_NUM, "2026-10-12 16:50:00", "DEBIT", "Transfer", 6001.0, "Transfer to Ramesh", 18501.0, 12500.0, "SUCCESS"),  # Transfer
        ]  # End of sample history
        cursor.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_history)  # Insert
    # Seed pending utility bills if empty
    cursor.execute("SELECT COUNT(*) FROM utility_bills WHERE account_number = ?", (DEFAULT_ACCOUNT_NUM,))  # Check bills
    if cursor.fetchone()[0] == 0:  # If no utility bills exist
        sample_bills = [  # List of realistic household bills
            (DEFAULT_ACCOUNT_NUM, "Electricity", "Mahavitaran (MSEB)", "CONS-8820194", 1200.0, "2026-10-25", "UNPAID"),  # Power bill
            (DEFAULT_ACCOUNT_NUM, "Water", "Municipal Water Board", "WTR-119283", 350.0, "2026-10-28", "UNPAID"),  # Water bill
            (DEFAULT_ACCOUNT_NUM, "Gas", "Mahanagar Gas (MGL)", "GAS-449102", 980.0, "2026-10-30", "UNPAID"),  # Gas bill
        ]  # End of sample bills
        cursor.executemany("INSERT INTO utility_bills (account_number, bill_type, provider_name, consumer_id, amount, due_date, status) VALUES (?, ?, ?, ?, ?, ?, ?)", sample_bills)  # Insert
    # Seed nearby ATM locations if empty
    cursor.execute("SELECT COUNT(*) FROM atm_locations")  # Check ATM locations
    if cursor.fetchone()[0] == 0:  # If no ATM entries exist
        sample_atms = [  # List of nearby ATMs with distances and accessibility info
            ("State Bank of India ATM", "Near Station Road, Opposite Post Office", 200, "Cash Available (₹500 & ₹100)", "Yes (Ramp available)"),  # SBI
            ("HDFC Bank Branch & ATM", "Shop 4, Ground Floor, Central Market", 450, "Cash Available (₹500)", "Yes"),  # HDFC
            ("Bank of Baroda ATM", "Near City Garden, Main Road", 800, "Cash Available", "No (3 steps)"),  # BOB
        ]  # End of sample ATMs
        cursor.executemany("INSERT INTO atm_locations (name, address, distance_meters, cash_status, wheelchair_access) VALUES (?, ?, ?, ?, ?)", sample_atms)  # Insert
    conn.commit()  # Commit all table schemas and seeded records to disk
    conn.close()  # Close database connection cleanly

# Initialize database tables and seed records at module load time                 # Description comment
init_banking_database()  # Run database initializer

# Helper function to generate unique bank transaction reference number           # Description comment
def generate_transaction_id() -> str:  # Define transaction ID generator function
    timestamp_part = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Get current timestamp digits
    random_part = random.randint(100, 999)  # Generate 3-digit random number
    return f"TXN{timestamp_part}{random_part}"  # Return formatted transaction reference string

# Service 1: Balance Enquiry implementation                                      # Description comment
def check_balance(account_number: str = DEFAULT_ACCOUNT_NUM) -> Dict[str, Any]:  # Define balance checker
    conn = get_db_connection()  # Connect to database
    cursor = conn.cursor()  # Get cursor
    cursor.execute("SELECT holder_name, balance, account_type, account_number FROM accounts WHERE account_number = ?", (account_number,))  # Query
    row = cursor.fetchone()  # Fetch account record
    conn.close()  # Close database connection
    if not row:  # Check if account was not found
        return {"status": "error", "message": "Account not found"}  # Return error
    balance = float(row["balance"])  # Extract current balance
    holder = row["holder_name"]  # Extract holder name
    acc_masked = "XXXX-" + account_number[-4:]  # Mask account number for security
    speech = f"Namaste {holder}. Aapke {row['account_type']} mein kul balance {balance:,.2f} rupaye hai."  # Hindi response
    display = (  # Build clean visual display card
        "\n" + "=" * 65 + "\n"
        "                  BANK ACCOUNT BALANCE ENQUIRY\n"
        "=" * 65 + "\n"
        f" Account Holder   : {holder}\n"
        f" Account Number   : {acc_masked}\n"
        f" Account Type     : {row['account_type']}\n"
        f" Available Balance: ₹{balance:,.2f}\n"
        "=" * 65 + "\n"
    )  # End of display card
    return {  # Return structured balance result
        "status": "success", "balance": balance, "holder": holder,  # Metadata
        "speech": speech, "display": display  # Outputs
    }  # End of dictionary

# Service 2: Fund Transfer with Balance BEFORE & AFTER display                   # Description comment
def transfer_funds(recipient_name: str, amount: float, account_number: str = DEFAULT_ACCOUNT_NUM) -> Dict[str, Any]:  # Define transfer function
    if amount <= 0:  # Validate positive amount
        return {"status": "error", "message": "Transaction amount must be greater than zero."}  # Return validation error
    conn = get_db_connection()  # Connect to database
    cursor = conn.cursor()  # Get cursor
    cursor.execute("SELECT balance, holder_name FROM accounts WHERE account_number = ?", (account_number,))  # Fetch sender balance
    account = cursor.fetchone()  # Retrieve row
    if not account:  # If account missing
        conn.close()  # Close connection
        return {"status": "error", "message": "Sender account not found."}  # Return error
    balance_before = float(account["balance"])  # Record balance BEFORE transfer
    if balance_before < amount:  # Check for insufficient funds
        conn.close()  # Close connection
        speech = f"Maafi kijiye! Aapke khate mein paryapt balance nahi hai. Aapka vartamaan balance {balance_before:,.2f} rupaye hai."  # Message
        return {  # Return failure dictionary
            "status": "insufficient_funds", "balance_before": balance_before,  # Balance before
            "speech": speech, "message": "Insufficient balance for this transfer."  # Details
        }  # End of return
    balance_after = balance_before - amount  # Calculate balance AFTER transfer
    txn_id = generate_transaction_id()  # Generate unique transaction ID
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current date and time
    # Deduct balance from sender account
    cursor.execute("UPDATE accounts SET balance = ? WHERE account_number = ?", (balance_after, account_number))  # Update balance
    # Record transaction entry in ledger
    cursor.execute("""
    INSERT INTO transactions (txn_id, account_number, timestamp, type, category, amount, details, balance_before, balance_after, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (txn_id, account_number, now_str, "DEBIT", "Transfer", amount, f"Transfer to {recipient_name}", balance_before, balance_after, "SUCCESS"))  # Insert
    conn.commit()  # Save database changes
    conn.close()  # Close connection
    speech = f"{recipient_name} ko {amount:,.0f} rupaye bhej diye gaye hain. Aapka purana balance {balance_before:,.2f} rupaye tha, ab naya balance {balance_after:,.2f} rupaye hai."  # Voice summary
    display = (  # Build formal transaction receipt card
        "\n" + "=" * 65 + "\n"
        "             FUNDS TRANSFER RECEIPT (COMPLETED)\n"
        "=" * 65 + "\n"
        f" Transaction ID   : {txn_id}\n"
        f" Date & Time      : {now_str}\n"
        f" Beneficiary Name : {recipient_name}\n"
        f" Transferred Amount: ₹{amount:,.2f}\n"
        "-" * 65 + "\n"
        f" Balance BEFORE   : ₹{balance_before:,.2f}\n"
        f" Balance AFTER    : ₹{balance_after:,.2f}  (Debited)\n"
        f" Status           : SUCCESSFUL ✓\n"
        "=" * 65 + "\n"
    )  # End of receipt card
    return {  # Return success result
        "status": "success", "txn_id": txn_id, "amount": amount, "recipient": recipient_name,  # Transfer details
        "balance_before": balance_before, "balance_after": balance_after,  # Before and after balance
        "speech": speech, "display": display  # Output representations
    }  # End of dictionary

# Service 3: Transaction History / Passbook Viewer                               # Description comment
def get_transaction_history(limit: int = 5, account_number: str = DEFAULT_ACCOUNT_NUM) -> Dict[str, Any]:  # Define history retriever
    conn = get_db_connection()  # Connect to database
    cursor = conn.cursor()  # Get cursor
    cursor.execute("""
    SELECT txn_id, timestamp, type, category, amount, details, balance_after
    FROM transactions WHERE account_number = ?
    ORDER BY timestamp DESC LIMIT ?
    """, (account_number, limit))  # Fetch latest transactions
    rows = cursor.fetchall()  # Get rows
    conn.close()  # Close connection
    if not rows:  # If no transactions
        return {"status": "success", "speech": "Khate mein koi pichhla transaction nahi mila.", "display": "No transactions found."}  # Empty return
    table_lines = [  # Build passbook ASCII table
        "\n" + "=" * 75,  # Top border
        f"                 ACCOUNT PASSBOOK: LAST {len(rows)} TRANSACTIONS",  # Title
        "=" * 75,  # Separator
        f" {'Date & Time':18} | {'Type':6} | {'Amount (₹)':10} | {'Details':25} | {'Balance (₹)':10}",  # Header
        "-" * 75  # Divider
    ]  # End of initial table lines
    speech_summary_parts = []  # Parts for spoken voice output
    for idx, r in enumerate(rows):  # Loop through rows
        date_short = r["timestamp"][:16]  # Shorten timestamp
        sign = "+" if r["type"] == "CREDIT" else "-"  # Sign prefix
        amt_str = f"{sign}₹{r['amount']:,.2f}"  # Format amount
        bal_str = f"₹{r['balance_after']:,.2f}"  # Format balance
        details = (r["details"][:24] + "..") if len(r["details"]) > 25 else r["details"]  # Truncate details
        table_lines.append(f" {date_short:18} | {r['type']:6} | {amt_str:10} | {details:25} | {bal_str:10}")  # Add row
        if idx < 3:  # Speak top 3 for brevity
            speech_summary_parts.append(f"{r['details']} {r['amount']:,.0f} rupaye {r['type'].lower()}")  # Add speech item
    table_lines.append("=" * 75 + "\n")  # Add bottom border
    display = "\n".join(table_lines)  # Combine into single text display
    speech = f"Aapke pichhle {len(rows)} transactions hain: " + ", ".join(speech_summary_parts) + "."  # Hindi voice summary
    return {"status": "success", "count": len(rows), "speech": speech, "display": display}  # Return dictionary

# Service 4: Mobile Recharge with Balance BEFORE & AFTER                          # Description comment
def execute_mobile_recharge(amount: float, phone: str = "9876543210", account_number: str = DEFAULT_ACCOUNT_NUM) -> Dict[str, Any]:  # Define recharge
    conn = get_db_connection()  # Connect to database
    cursor = conn.cursor()  # Get cursor
    cursor.execute("SELECT balance FROM accounts WHERE account_number = ?", (account_number,))  # Fetch balance
    account = cursor.fetchone()  # Retrieve row
    if not account:  # If account missing
        conn.close()  # Close connection
        return {"status": "error", "message": "Account not found"}  # Return error
    balance_before = float(account["balance"])  # Balance BEFORE recharge
    if balance_before < amount:  # Insufficient balance check
        conn.close()  # Close connection
        speech = f"Recharge ke liye paryapt balance nahi hai. Aapka balance {balance_before:,.2f} rupaye hai."  # Insufficient speech
        return {"status": "insufficient_funds", "speech": speech, "message": "Insufficient balance"}  # Return failure
    balance_after = balance_before - amount  # Balance AFTER recharge
    txn_id = generate_transaction_id()  # Transaction ID
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
    cursor.execute("UPDATE accounts SET balance = ? WHERE account_number = ?", (balance_after, account_number))  # Update balance
    cursor.execute("""
    INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (txn_id, account_number, now_str, "DEBIT", "Recharge", amount, f"Prepaid Recharge: {phone}", balance_before, balance_after, "SUCCESS"))  # Ledger
    conn.commit()  # Save
    conn.close()  # Close
    speech = f"Mobile number {phone} par {amount:,.0f} rupaye ka recharge safal raha. Purana balance {balance_before:,.2f} tha, naya balance {balance_after:,.2f} rupaye hai."  # Speech
    display = (  # Receipt display
        "\n" + "=" * 65 + "\n"
        "              MOBILE RECHARGE RECEIPT\n"
        "=" * 65 + "\n"
        f" Mobile Number    : {phone}\n"
        f" Recharge Amount  : ₹{amount:,.2f}\n"
        f" Transaction ID   : {txn_id}\n"
        "-" * 65 + "\n"
        f" Balance BEFORE   : ₹{balance_before:,.2f}\n"
        f" Balance AFTER    : ₹{balance_after:,.2f}\n"
        f" Status           : SUCCESSFUL ✓\n"
        "=" * 65 + "\n"
    )  # End of display
    return {"status": "success", "balance_before": balance_before, "balance_after": balance_after, "speech": speech, "display": display}  # Return

# Service 5: Utility Bill Payment with Balance BEFORE & AFTER                    # Description comment
def pay_utility_bill(bill_type: str = "Electricity", amount: Optional[float] = None, account_number: str = DEFAULT_ACCOUNT_NUM) -> Dict[str, Any]:  # Define bill payment
    conn = get_db_connection()  # Connect to database
    cursor = conn.cursor()  # Get cursor
    # Check if a bill record exists for this type
    cursor.execute("SELECT provider_name, amount, consumer_id FROM utility_bills WHERE account_number = ? AND bill_type LIKE ? LIMIT 1", (account_number, f"%{bill_type}%"))  # Query bill
    bill_row = cursor.fetchone()  # Retrieve bill row
    provider = bill_row["provider_name"] if bill_row else f"{bill_type} Board"  # Provider name
    bill_amount = amount if amount is not None else (float(bill_row["amount"]) if bill_row else 1200.0)  # Bill amount
    cursor.execute("SELECT balance FROM accounts WHERE account_number = ?", (account_number,))  # Fetch balance
    account = cursor.fetchone()  # Retrieve row
    balance_before = float(account["balance"])  # Balance BEFORE payment
    if balance_before < bill_amount:  # Insufficient funds check
        conn.close()  # Close connection
        speech = f"Bill payment ke liye paryapt balance nahi hai. Aapka balance {balance_before:,.2f} rupaye hai."  # Insufficient speech
        return {"status": "insufficient_funds", "speech": speech, "message": "Insufficient balance"}  # Return failure
    balance_after = balance_before - bill_amount  # Balance AFTER payment
    txn_id = generate_transaction_id()  # Generate TXN ID
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current time
    cursor.execute("UPDATE accounts SET balance = ? WHERE account_number = ?", (balance_after, account_number))  # Deduct balance
    cursor.execute("UPDATE utility_bills SET status = 'PAID' WHERE account_number = ? AND bill_type LIKE ?", (account_number, f"%{bill_type}%"))  # Mark bill paid
    cursor.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (txn_id, account_number, now_str, "DEBIT", "Bill Payment", bill_amount, f"{bill_type} Bill: {provider}", balance_before, balance_after, "SUCCESS"))  # Ledger
    conn.commit()  # Commit
    conn.close()  # Close
    speech = f"{provider} ka {bill_type} bill {bill_amount:,.0f} rupaye safalta se bhar diya gaya hai. Purana balance {balance_before:,.2f} tha, naya balance {balance_after:,.2f} rupaye hai."  # Speech
    display = (  # Receipt display
        "\n" + "=" * 65 + "\n"
        f"             UTILITY BILL PAYMENT RECEIPT\n"
        "=" * 65 + "\n"
        f" Bill Category    : {bill_type}\n"
        f" Biller Provider  : {provider}\n"
        f" Bill Amount Paid : ₹{bill_amount:,.2f}\n"
        f" Transaction ID   : {txn_id}\n"
        "-" * 65 + "\n"
        f" Balance BEFORE   : ₹{balance_before:,.2f}\n"
        f" Balance AFTER    : ₹{balance_after:,.2f}\n"
        f" Status           : PAID & COMPLETED ✓\n"
        "=" * 65 + "\n"
    )  # End of receipt
    return {"status": "success", "balance_before": balance_before, "balance_after": balance_after, "speech": speech, "display": display}  # Return

# Service 6: ATM and Branch Locator                                              # Description comment
def locate_nearest_atms() -> Dict[str, Any]:  # Define ATM locator function
    conn = get_db_connection()  # Connect to database
    cursor = conn.cursor()  # Get cursor
    cursor.execute("SELECT name, address, distance_meters, cash_status, wheelchair_access FROM atm_locations ORDER BY distance_meters ASC")  # Fetch ATMs
    rows = cursor.fetchall()  # Retrieve all rows
    conn.close()  # Close connection
    lines = [  # Build display
        "\n" + "=" * 70 + "\n"
        "                  NEAREST ATMS & BRANCHES\n"
        "=" * 70
    ]  # End header
    for r in rows:  # Loop ATMs
        dist_str = f"{r['distance_meters']} meters away" if r['distance_meters'] < 1000 else f"{r['distance_meters']/1000:.1f} km away"  # Format distance
        lines.append(f" 📍 {r['name']} ({dist_str})")  # Name and distance
        lines.append(f"    Address    : {r['address']}")  # Address
        lines.append(f"    Cash Status: {r['cash_status']}")  # Cash status
        lines.append(f"    Accessibility: {r['wheelchair_access']}")  # Wheelchair
        lines.append("-" * 70)  # Separator
    display = "\n".join(lines) + "\n"  # Final display text
    first_atm = rows[0]  # Nearest ATM
    speech = f"Sabse paas ka ATM hai {first_atm['name']}, jo lagbhag {first_atm['distance_meters']} meter door hai, aur cash uplabdh hai."  # Hindi speech
    return {"status": "success", "count": len(rows), "speech": speech, "display": display}  # Return

# Service 7: Help & Instant Debit Card Blocking                                  # Description comment
def handle_help_and_security(action_type: str = "general", account_number: str = DEFAULT_ACCOUNT_NUM) -> Dict[str, Any]:  # Define help handler
    conn = get_db_connection()  # Connect to database
    cursor = conn.cursor()  # Get cursor
    if "block" in action_type.lower() or "kho gaya" in action_type.lower() or "lost" in action_type.lower():  # Check if card blocking requested
        cursor.execute("UPDATE accounts SET card_status = 'BLOCKED' WHERE account_number = ?", (account_number,))  # Block card
        cursor.execute("SELECT card_number FROM accounts WHERE account_number = ?", (account_number,))  # Get card number
        card_num = cursor.fetchone()["card_number"]  # Card string
        conn.commit()  # Save
        conn.close()  # Close
        speech = f"Suraksha ke liye aapka Debit Card turant block kar diya gaya hai. Kripya naye card ke liye helpline 1800 120 4000 par sampark karein."  # Speech
        display = (  # Display card
            "\n" + "=" * 65 + "\n"
            "         EMERGENCY SECURITY ACTION: DEBIT CARD BLOCKED\n"
            "=" * 65 + "\n"
            f" Card Number      : {card_num}\n"
            f" Current Status   : BLOCKED (HOTLISTED) 🔒\n"
            f" Reason           : User reported lost / fraud concern\n"
            f" 24x7 Helpline    : 1800-120-4000 (Toll-Free)\n"
            "=" * 65 + "\n"
        )  # End of display
        return {"status": "card_blocked", "speech": speech, "display": display}  # Return
    conn.close()  # Close connection
    speech = "Aap SIO 24 ghante customer helpline 1800 120 4000 par call kar sakte hain. Bank sahayak aapki poori madad karenge."  # General help speech
    display = (  # Help display
        "\n" + "=" * 65 + "\n"
        "                  SIO CUSTOMER ASSISTANCE & HELP\n"
        "=" * 65 + "\n"
        " Toll-Free Helpline: 1800-120-4000 (Available 24x7)\n"
        " Senior Citizen Care: Press 1 for Senior Priority Desk\n"
        " Email Support     : support@siobank.in\n"
        " Emergency Services: Block Card, Report Fraud, Stop Cheque\n"
        "=" * 65 + "\n"
    )  # End of display
    return {"status": "help_provided", "speech": speech, "display": display}  # Return

# Unified execution entry point: maps Dialogue Manager output directly to banking # Description comment
def execute_banking_action(dialogue_result: Dict[str, Any], account_number: str = DEFAULT_ACCOUNT_NUM) -> Dict[str, Any]:  # Define unified executor
    intent = dialogue_result.get("intent", "Help")  # Get intent
    entities = dialogue_result.get("entities", {})  # Get entities
    if intent == "Balance Enquiry":  # Check balance
        return check_balance(account_number)  # Call balance
    elif intent == "Fund Transfer":  # Check transfer
        recipient = entities.get("recipient", "Recipient")  # Recipient
        amount = float(entities.get("amount", 0))  # Amount
        return transfer_funds(recipient, amount, account_number)  # Call transfer
    elif intent == "Transaction History":  # Check history
        return get_transaction_history(5, account_number)  # Call history
    elif intent == "Mobile Recharge":  # Check recharge
        amount = float(entities.get("amount", 199.0))  # Amount
        return execute_mobile_recharge(amount, "9876543210", account_number)  # Call recharge
    elif intent == "Bill Payment":  # Check bill
        bill_type = entities.get("bill_type", "Electricity")  # Bill type
        amount = entities.get("amount")  # Optional amount
        amt_float = float(amount) if amount else None  # Parse amount
        return pay_utility_bill(bill_type, amt_float, account_number)  # Call bill
    elif intent == "ATM Locator":  # Check ATM
        return locate_nearest_atms()  # Call ATM
    elif intent == "Help":  # Check help
        query_text = dialogue_result.get("message", "")  # Get message text
        return handle_help_and_security(query_text, account_number)  # Call help
    else:  # Unrecognized fallback
        return {"status": "unknown", "speech": "Kaam poora nahi ho saka.", "display": "Unsupported service request."}  # Fallback
