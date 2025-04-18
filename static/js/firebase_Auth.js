// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getDatabase, set, ref, get, update } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-database.js";
import { getAuth, createUserWithEmailAndPassword, signOut, signInWithEmailAndPassword, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

// Firebase configuration
const firebaseConfig = window.firebaseConfig;
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);
const auth = getAuth(app);

// Function to show Bootstrap modal messages
function showModal(message, type = "info", redirectUrl = null) {
    const modalMessage = document.getElementById("modalMessage");
    modalMessage.textContent = message;

    // Show the Bootstrap modal
    const modal = new bootstrap.Modal(document.getElementById("messageModal"));
    modal.show();

    // Auto-close modal after 3 seconds and redirect if needed
    setTimeout(() => {
        modal.hide();
        if (redirectUrl) {
            window.location.href = redirectUrl;
        }
    }, 3000);
}

// Wait for the DOM to load
window.addEventListener('load', () => {
    const signupButton = document.getElementById('Signup');
    if (signupButton) {
        signupButton.addEventListener('click', async (e) => {
            e.preventDefault();

            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value.trim();
            const confirmPassword = document.getElementById('confirm_password')?.value.trim();
            const username = document.getElementById('username').value.trim();

            const barCouncilField = document.getElementById('barCouncilID');
            const barCouncilID = barCouncilField ? barCouncilField.value.trim() : null;
            const isLawyer = barCouncilID !== null && barCouncilID !== "";

            if (!email || !password || !username) {
                showModal("Please fill in all fields.", "error");
                return;
            }

            if (confirmPassword && password !== confirmPassword) {
                showModal("Passwords do not match.", "error");
                return;
            }

            try {
                const userTypeRef = isLawyer ? ref(database, 'lawyers') : ref(database, 'users');
                const snapshot = await get(userTypeRef);

                if (snapshot.exists()) {
                    const users = snapshot.val();
                    for (const uid in users) {
                        if (users[uid].email === email) {
                            showModal("Email is already registered. Try logging in.", "error");
                            return;
                        }
                        if (isLawyer && users[uid].barCouncilID === barCouncilID) {
                            showModal("Bar Council ID is already in use.", "error");
                            return;
                        }
                    }
                }

                const userCredential = await createUserWithEmailAndPassword(auth, email, password);
                const user = userCredential.user;

                const userData = { username, email };
                if (isLawyer) userData.barCouncilID = barCouncilID;

                const dbPath = isLawyer ? `lawyers/${user.uid}` : `users/${user.uid}`;
                await set(ref(database, dbPath), userData);

                console.log("User data saved successfully.");
                showModal("Signup successful! Redirecting...", "success", afterloginUrl);

            } catch (error) {
                console.error("Error during signup:", error.message);
                showModal(error.message, "error");
            }
        });
    }

    const loginButton = document.getElementById('login');
    if (loginButton) {
        loginButton.addEventListener('click', async (e) => {
            e.preventDefault();

            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value.trim();

            try {
                const userCredential = await signInWithEmailAndPassword(auth, email, password);
                const user = userCredential.user;

                const dt = new Date();
                await update(ref(database, `users/${user.uid}`), { last_login: dt });

                localStorage.setItem('clientEmail', email);

                console.log("User logged in.");
                showModal("Login successful! Redirecting...", "success", afterloginUrl);

            } catch (error) {
                console.error("Login error:", error.message);
                showModal(error.message, "error");
            }
        });
    }

    const logoutButton = document.getElementById('logout');
    if (logoutButton) {
        logoutButton.addEventListener('click', async () => {
            try {
                await signOut(auth);
                console.log("User logged out.");
                showModal("Logged out successfully! Redirecting...", "success", indexUrl);
            } catch (error) {
                console.error("Logout error:", error.message);
                showModal(error.message, "error");
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