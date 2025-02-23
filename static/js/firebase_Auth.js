// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getDatabase, set, ref, get } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-database.js";
import { getAuth, createUserWithEmailAndPassword, signOut, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

// Firebase configuration
const firebaseConfig = window.firebaseConfig;
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);
const auth = getAuth(app);

// Wait for the DOM to load
window.addEventListener('load', () => {
  const signupButton = document.getElementById('signup');
  if (signupButton) {
    signupButton.addEventListener('click', async (e) => {
      e.preventDefault(); // Prevent form submission

      const email = document.getElementById('email').value.trim();
      const password = document.getElementById('password').value.trim();
      const confirmPassword = document.querySelectorAll('.pass-key')[1].value.trim();
      const username = document.getElementById('username').value.trim();
      const barCouncilID = document.getElementById('barCouncilID').value.trim(); // Ensure Bar Council ID field exists

      if (!email || !password || !username || !barCouncilID) {
        alert("Please fill in all fields.");
        return;
      }

      if (password !== confirmPassword) {
        alert("Passwords do not match.");
        return;
      }

      try {
        // **Step 1: Check if the Bar Council ID is already used**
        const barCouncilRef = ref(database, 'users');
        const snapshot = await get(barCouncilRef);

        if (snapshot.exists()) {
          const users = snapshot.val();
          for (const uid in users) {
            if (users[uid].email === email) {
              alert("Email is already registered. Try logging in.");
              return;
            }
            if (users[uid].barCouncilID === barCouncilID) {
              alert("Bar Council ID is already in use.");
              return;
            }
          }
        }

        // **Step 2: Create user in Firebase Authentication**
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // **Step 3: Store user data in Firebase Realtime Database**
        await set(ref(database, `users/${user.uid}`), {
          username: username,
          email: email,
          barCouncilID: barCouncilID,
        });

        console.log("User data saved successfully.");
        
        // Redirect after successful signup
        window.location.href = afterloginUrl;

      } catch (error) {
        console.error("Error during signup:", error.message);
        alert(error.message);
      }
    });
  }

  // Logout event listener
  const logoutButton = document.getElementById('logout');
  if (logoutButton) {
    logoutButton.addEventListener('click', async () => {
      try {
        await signOut(auth);
        console.log("User logged out.");
        window.location.href = indexUrl; // Redirect to home/login page
      } catch (error) {
        console.error("Logout error:", error.message);
        alert(error.message);
      }
    });
  }
});

// Listen for authentication state changes
onAuthStateChanged(auth, (user) => {
  if (user) {
    console.log('User signed in:', user.uid);
  } else {
    console.log('User signed out.');
  }
});
