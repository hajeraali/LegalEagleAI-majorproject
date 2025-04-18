// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getDatabase, ref, set, get } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-database.js";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

// Firebase configuration
const firebaseConfig = window.firebaseConfig;
// Initialize Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);
const auth = getAuth(app);

// Load lawyer data from the uploaded dataset
let lawyerData = [];

fetch('static/js/LawyerDataForAuth.csv')
  .then(response => response.text())
  .then(data => {
    lawyerData = data.split("\n").slice(1).map(row => {
      const [Sr_No, Lawyer_name, Bar_Council_ID] = row.split(",");
      return { Lawyer_name: Lawyer_name.trim(), Bar_Council_ID: Bar_Council_ID.trim() };
    });
    console.log("Lawyer data loaded:", lawyerData); // Debugging output
  })
  .catch(error => console.error("Error loading lawyer data:", error));

// Function to check if a lawyer's name and Bar Council ID match
function validateLawyerDetails(name, id) {
  const match = lawyerData.find(lawyer => lawyer.Lawyer_name === name && lawyer.Bar_Council_ID === id);
  return match ? { found: true, status: true } : { found: false, status: false };
}

// Wait for the DOM to load
window.addEventListener('load', () => {
  // Signup button event listener
  const signupButton = document.getElementById('Signup');
  if (signupButton) {
    signupButton.addEventListener('click', (e) => {
      e.preventDefault(); // Prevent form submission

      const name = document.getElementById('fullName').value.trim();
      const barCouncilID = document.getElementById('barCouncilID').value.trim();
      const email = document.getElementById('email').value.trim();
      const password = document.getElementById('password').value;
      const confirmPassword = document.getElementById('confirmPassword').value;

      // Check if passwords match
      if (password !== confirmPassword) {
        alert("Passwords do not match. Please try again.");
        return;
      }

      // Validate the lawyer's name and Bar Council ID
      const lawyerCheck = validateLawyerDetails(name, barCouncilID);
      if (!lawyerCheck.found) {
        alert("Invalid Name or Bar Council ID. Registration allowed, but your account will be pending approval.");
      }

      // Proceed with Firebase signup
      createUserWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
          const user = userCredential.user;
          console.log("User created:", user); // Debugging output

          set(ref(database, 'lawyers/' + user.uid), {
            name: name,
            barCouncilID: barCouncilID,
            email: email,
            status: lawyerCheck.status // If found in CSV: true, otherwise: false
          })
          .then(() => {
            localStorage.setItem('lawyerBarCouncilID', barCouncilID);
            console.log("Data successfully written to the database");
    
            if (lawyerCheck.status) {
                alert("Registration successful. Redirecting to dashboard...");
                window.location.href = dashboardUrl; // Approved lawyers go to the dashboard
            } else {
                alert("Registration successful but pending admin approval. Redirecting to homepage.");
                window.location.href = indexUrl; // Unapproved lawyers go to the homepage
            }
          })
          .catch((error) => {
            console.error("Error writing data to the database:", error);
            alert("Error writing to the database: " + error.message);
          });
        })
        .catch((error) => {
          console.error("Error during signup:", error);
          alert("Error: " + error.message);
        });
    });
  } else {
    console.error("Signup button not found");
  }

  // Login logic
  const loginButton = document.getElementById('login');
  if (loginButton) {
    loginButton.addEventListener('click', (e) => {
      e.preventDefault(); // Prevent form submission

      const email = document.getElementById('loginEmail').value.trim();
      const barCouncilID = document.getElementById('loginBarCouncilID').value.trim();
      const password = document.getElementById('loginPassword').value;

      // Firebase login
      signInWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
          const user = userCredential.user;
          console.log("User logged in:", user); // Debugging output

          // Check Bar Council ID in the database
          const userRef = ref(database, 'lawyers/' + user.uid);
          get(userRef).then((snapshot) => {
            if (snapshot.exists()) {
              const data = snapshot.val();
              if (data.barCouncilID === barCouncilID) {
                localStorage.setItem('lawyerName', data.name);
                localStorage.setItem('lawyerBarCouncilID', data.barCouncilID);
                if (data.status) {
                  // Redirect to the dashboard if approved
                  window.location.href = dashboardUrl;
                } else {
                  // Redirect to index page if not yet approved
                  alert("Your registration is pending admin approval. Redirecting to the home page.");
                  window.location.href = indexUrl; // Change this to your actual index page URL
                }
              } else {
                alert("Bar Council ID does not match. Please try again.");
              }
            } else {
              alert("User data not found. Please check your details.");
            }
          }).catch((error) => {
            console.error("Error fetching user data:", error);
            alert("Error fetching user data: " + error.message);
          });
        })
        .catch((error) => {
          console.error("Error during login:", error);
          alert("Error: " + error.message);
        });
    });
  } else {
    console.error("Login button not found");
  }
});