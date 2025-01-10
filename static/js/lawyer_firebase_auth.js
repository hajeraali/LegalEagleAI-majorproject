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
  return lawyerData.some(lawyer => lawyer.Lawyer_name === name && lawyer.Bar_Council_ID === id);
}

// Wait for the DOM to load
window.addEventListener('load', () => {
  // Signup button event listener
  const signupButton = document.getElementById('Signup');
  if (signupButton) {
    signupButton.addEventListener('click', (e) => {
      e.preventDefault(); // Prevent form submission

      const name = document.getElementById('fullName').value;
      const barCouncilID = document.getElementById('barCouncilID').value;
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      const confirmPassword = document.getElementById('confirmPassword').value;

      // Check if passwords match
      if (password !== confirmPassword) {
        alert("Passwords do not match. Please try again.");
        return;
      }

      // Validate the lawyer's name and Bar Council ID
      if (!validateLawyerDetails(name, barCouncilID)) {
        alert("Invalid Name or Bar Council ID. Please check your details.");
        return;
      }

      // Proceed with Firebase signup
      createUserWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
          const user = userCredential.user;
          console.log("User created:", user); // Debugging output

          set(ref(database, 'lawyers/' + user.uid), {
            name: name,
            barCouncilID: barCouncilID,
            email: email
          })
          .then(() => {
            console.log("Data successfully written to the database");
            window.location.href = dashboardUrl; // Redirect to the desired page
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

      const email = document.getElementById('loginEmail').value;
      const barCouncilID = document.getElementById('loginBarCouncilID').value;
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
                window.location.href = dashboardUrl; // Redirect to the desired page
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
