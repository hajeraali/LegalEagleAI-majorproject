// Import the functions you need from the Firebase SDK
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getDatabase, set, ref, update } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-database.js";
import { getAuth, createUserWithEmailAndPassword, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

// Define firebaseConfig in the global scope
let firebaseConfig;

// Fetch the Firebase configuration from the Flask backend
fetch('/get-firebase-config')
  .then(response => response.json())
  .then(data => {
    // Initialize Firebase with the configuration returned from the server
    firebaseConfig = {
      apiKey: data.apiKey,
      authDomain: data.authDomain,
      projectId: data.projectId,
      storageBucket: data.storageBucket,
      messagingSenderId: data.messagingSenderId,
      appId: data.appId
    };

    // Initialize Firebase
    initializeApp(firebaseConfig);
  })
  .catch(error => {
    console.error('Error fetching Firebase configuration:', error);
  });

// Initialize Firebase services outside the fetch call, but after firebaseConfig is set
const app = initializeApp(firebaseConfig);  // This will fail if firebaseConfig isn't set yet, so make sure fetch is done first
const database = getDatabase(app);
const auth = getAuth(app);

// Wait for the DOM to load
window.addEventListener('load', () => {
  // Signup button event listener
  const signupButton = document.getElementById('signup');
  if (signupButton) {
    signupButton.addEventListener('click', (e) => {
      e.preventDefault(); // Prevent form submission
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      const confirmPassword = document.querySelectorAll('.pass-key')[1].value; // Selecting Confirm Password field
      const username = document.getElementById('username').value;

      // Check if passwords match
      if (password !== confirmPassword) {
        alert("Passwords do not match. Please try again.");
        return;
      }

      // Firebase signup
      createUserWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
          const user = userCredential.user;
          set(ref(database, 'users/' + user.uid), {
            username: username,
            email: email
          });
          // Redirect to a new page after signup
          window.location.href = afterloginUrl; // Change "welcome.html" to your desired page
        })
        .catch((error) => {
          alert(error.message);
        });
    });
  }

  // Logout button event listener
  const logoutButton = document.getElementById('logout');
  if (logoutButton) {
    logoutButton.addEventListener('click', () => {
      signOut(auth)
        .then(() => {
          // Redirect to a login or home page after logout
          window.location.href = indexUrl;
        })
        .catch((error) => {
          alert(error.message);
        });
    });
  }
});

// Auth state change listener
onAuthStateChanged(auth, (user) => {
  if (user) {
    console.log('User is signed in:', user.uid);
  } else {
    console.log('User is signed out');
  }
});
